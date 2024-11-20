import torchtext
import gc
import psutil
import shutil
import os
import onnx
import torch
import torch.nn as nn
import math
import copy
import time
import torchtext.datasets as datasets
import spacy
import warnings 
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.functional import log_softmax, pad
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from os.path import exists

from batch import Batch
from model import make_model
from functools import partial
from multiprocessing import Pool
from label_smoothing import LabelSmoothing
from onnx.shape_inference import infer_shapes
from onnx_optimized_inference import run_module
from qonnx.core.modelwrapper import ModelWrapper
import nltk
import onnxruntime as ort
import numpy as np
import copy
import json
import onnx.numpy_helper as numpy_helper

# Set to False to skip notebook execution (e.g. for debugging)
ort.set_default_logger_severity(3)
warnings.filterwarnings("ignore")
RUN_EXAMPLES = True
# Some convenience helper functions used throughout the notebook

import argparse
parser = argparse.ArgumentParser('Fault Injection Program')
parser.add_argument('--directory_name')
parser.add_argument('--experiment_output_name')
parser.add_argument('--module')
args = parser.parse_args()

from inject_utils.layers import *
from inject_utils.utils import *

def is_interactive_notebook():
    return __name__ == "__main__"

def show_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        return fn(*args)

def execute_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        fn(*args)

# Load spacy tokenizer models, download them if they haven't been
# downloaded already
def load_tokenizers():
    try:
        spacy_de = spacy.load("de_core_news_sm")
    except IOError:
        os.system("python -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")

    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    return spacy_de, spacy_en

def tokenize(text, tokenizer):
    #return [tok.text for tok in tokenizer.tokenizer(text)]
    temp = text.split(" ")
    #print(temp)
    return temp


def yield_tokens(data_iter, tokenizer, index):
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])

def build_vocabulary(spacy_de, spacy_en):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    print("Building German Vocabulary ...")
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    vocab_src = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_de, index=0),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    print("Building English Vocabulary ...")
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_en, index=1),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt


def load_vocab(spacy_de, spacy_en):
    if not exists("vocab.pt"):
        vocab_src, vocab_tgt = build_vocabulary(spacy_de, spacy_en)
        torch.save((vocab_src, vocab_tgt), "vocab.pt")
    else:
        vocab_src, vocab_tgt = torch.load("vocab.pt")
    print("Finished.\nVocabulary sizes:")
    print(len(vocab_src))
    print(len(vocab_tgt))
    return vocab_src, vocab_tgt


if is_interactive_notebook():
    #global variables used later in the script
    spacy_de, spacy_en = show_example(load_tokenizers)
    vocab_src, vocab_tgt = show_example(load_vocab, args=[spacy_de, spacy_en])

def collate_batch(
    batch,
    src_pipeline,
    tgt_pipeline,
    src_vocab,
    tgt_vocab,
    device,
    max_padding=128,
    pad_id=2,
):
    device="cpu"
    bs_id = torch.tensor([0], device=device)  # <s> token id
    eos_id = torch.tensor([1], device=device)  # </s> token id
    src_list, tgt_list = [], []
    for (_src, _tgt) in batch:
        processed_src = torch.cat(
            [
                bs_id,
                torch.tensor(
                    src_vocab(src_pipeline(_src)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        processed_tgt = torch.cat(
            [
                bs_id,
                torch.tensor(
                    tgt_vocab(tgt_pipeline(_tgt)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        src_list.append(
            # warning - overwrites values for negative values of padding - len
            pad(
                processed_src,
                (
                    0,
                    max_padding - len(processed_src),
                ),
                value=pad_id,
            )
        )
        tgt_list.append(
            pad(
                processed_tgt,
                (0, max_padding - len(processed_tgt)),
                value=pad_id,
            )
        )

    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)
    return (src, tgt)

#For standard
class TrainDatasets(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.datasets = list(dataset)

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        src, trg = self.datasets[idx]
        return src, trg

def create_dataset(source_bpe, target_bpe):
    corpus_source = open(source_bpe, 'r')
    corpus_target = open(target_bpe, 'r')

    lines_source = corpus_source.readlines()
    lines_target = corpus_target.readlines()

    sentence_pairs = []

    for source_sentence, target_sentence in zip(lines_source, lines_target):
        sentence_pairs.append((source_sentence[:-1], target_sentence[:-1]))

    return TrainDatasets(sentence_pairs)


def create_dataloaders(
    device,
    vocab_src,
    vocab_tgt,
    spacy_de,
    spacy_en,
    batch_size=12000,
    max_padding=72,
    is_distributed=True,
):
    # def create_dataloaders(batch_size=12000):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    def collate_fn(batch):
        return collate_batch(
            batch,
            tokenize_de,
            tokenize_en,
            vocab_src,
            vocab_tgt,
            device,
            max_padding=max_padding,
            pad_id=vocab_src.get_stoi()["<blank>"],
        )

    """
    train_iter, valid_iter, test_iter = datasets.Multi30k(
        language_pair=("de", "en")
    )
    """
    train_iter = create_dataset("./data/train.de.bpe","./data/train.en.bpe")
    valid_iter = create_dataset("./data/valid.de.bpe","./data/valid.en.bpe")
    test_iter = create_dataset("./data/test.de.bpe","./data/test.en.bpe")

    train_iter_map = to_map_style_dataset(
        train_iter
    )  # DistributedSampler needs a dataset len()
    train_sampler = (
        DistributedSampler(train_iter_map) if is_distributed else None
    )
    valid_iter_map = to_map_style_dataset(valid_iter)
    valid_sampler = (
        DistributedSampler(valid_iter_map) if is_distributed else None
    )

    train_dataloader = DataLoader(
        train_iter_map,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_iter_map,
        batch_size=batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler,
        collate_fn=collate_fn,
    )
    return train_dataloader, valid_dataloader


def fix_sentence(output_target):
    output_target = output_target.replace("@@ ","")
    output_target = output_target.replace("<s> ","")
    output_target = output_target.replace("</s>","")
    output_target = output_target.replace(" &apos;","'")
    output_target = output_target.split(" ")
    return output_target

def check_outputs(
    model_path,
    vocab_src,
    vocab_tgt,
    inject_parameters,
    n_examples,
    pad_idx,
    eos_string,
    batch,
):
    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device("cpu"))
    )
    results = [()] * n_examples
    reference_text = []
    output_text = []
    model.eval()

    fault_injection_results = []
    golden_and_faulty_results = []
    for idx in range(n_examples):
        #print("\nExample %d ========\n" % idx)
        #b = next(iter(valid_dataloader))
        b = batch[idx]
        rb = Batch(b[0], b[1], pad_idx)
        #greedy_decode(model, rb.src, rb.src_mask, 64, 0, False)[0]
        for inject_fault in [False, True]:

            src_tokens = [
                vocab_src.get_itos()[x] for x in rb.src[0] if x != pad_idx
            ]
            tgt_tokens = [
                vocab_tgt.get_itos()[x] for x in rb.tgt[0] if x != pad_idx
            ]

            if inject_fault:
                print("\nExample %d - Faulty ========\n" % idx)
            else:
                print("\nExample %d - Golden ========\n" % idx)

            """
            print(
                "Source Text (Input)        : "
                + " ".join(src_tokens).replace("\n", "")
            )
            print(
                "Target Text (Ground Truth) : "
                + " ".join(tgt_tokens).replace("\n", "")
            )
            """

            if inject_fault:
                model_out = greedy_decode(model, rb.src, rb.src_mask, 72, 0, True, inject_parameters)[0]
            else:
                model_out = greedy_decode(model, rb.src, rb.src_mask, 72, 0, False, None)[0]
            model_txt = (
                " ".join(
                    [vocab_tgt.get_itos()[x] for x in model_out if x != pad_idx]
                ).split(eos_string, 1)[0]
                + eos_string
            )

            tgt_tokens = " ".join(tgt_tokens).replace("\n", "")
            print("LIST:")
            output_target = tgt_tokens
            output_target = fix_sentence(output_target)
            output_target = output_target[:-1]
            reference_text.append([output_target])
            print(output_target)

            print("LIST:")
            output_list = [vocab_tgt.get_itos()[x] for x in model_out if x != pad_idx]
            end_index = (output_list.index("</s>"))
            output_list = output_list[1:end_index]
            output_string = (" ".join(output_list))
            output_list = fix_sentence(output_string)
            output_text.append(output_list)

            print(output_list)
            print("Model Output               : " + model_txt.replace("\n", ""))
            cc = nltk.translate.bleu_score.SmoothingFunction()

            print(nltk.translate.bleu_score.sentence_bleu([output_target], output_list, smoothing_function=cc.method4))
            bleu_score = nltk.translate.bleu_score.sentence_bleu([output_target], output_list, smoothing_function=cc.method4)
            golden_and_faulty_results.append(bleu_score)
            results[idx] = (rb, src_tokens, tgt_tokens, model_out, model_txt)

            if inject_fault:
                #print("CORPUS BLEU:")
                #print(nltk.translate.bleu_score.corpus_bleu(reference_text, output_text))
                print("COMPARE:")
                print(golden_and_faulty_results)
                with open(inject_parameters["experiment_output_file"], "a") as file:
                    file.write(str(str(inject_parameters["faulty_operation_name"]) + "," + str(golden_and_faulty_results[0]) + "," + str(golden_and_faulty_results[1]) + "," + str(inject_parameters["faulty_bit_position"]) + "\n"))
                golden_and_faulty_results = []
    #print(fault_injection_results)
    return fault_injection_results

def build_auxiliary_graphs(inject_parameters):
    def extract_model(input_path, output_path, target_input, input_names, output_names):
        onnx.utils.extract_model(input_path, ("./separated/" + output_path), input_names, output_names)

    os.system("rm ./separated/*")

    module = "decoder"
    input_names = ["global_in", "global_in_1", "global_in_2", "global_in_3"]
    original_input_names = ["global_in", "global_in_1", "global_in_2", "global_in_3"]
    if "Encoder" in inject_parameters["targetted_module"]:
        module = "encoder"
        input_names = ["global_in", "global_in_1"]
        original_input_names = ["global_in", "global_in_1"]
    input_path = "./try/" + module + "_try_cleaned.onnx"
    #output_path = "./separated/modified_module"

    output_names = ["global_out"]
    original_output_names = ["global_out"]

    if ("INPUT" in inject_parameters["inject_type"]) or ("WEIGHT" in inject_parameters["inject_type"]):
        target_input = inject_parameters["faulty_tensor_name"]
        output_names = [target_input]
        extract_model(input_path, "tensor_to_inject.onnx", None, input_names, output_names)

        input_names = input_names + [target_input]
        output_names = [inject_parameters["faulty_output_tensor"]]
        extract_model(input_path, "layer_to_inject.onnx", None, input_names, output_names)

    elif ("RANDOM" in inject_parameters["inject_type"]):
        input_names = input_names
        output_names = [inject_parameters["faulty_output_tensor"]]
        extract_model(input_path, "layer_to_inject.onnx", None, input_names, output_names)

    original_input_names = original_input_names + output_names
    extract_model(input_path, "rest_of_layers.onnx", None, original_input_names, original_output_names)

def run_model_example(model_path, inject_parameters, n_examples=5, number_of_parallelized_experiments=5):
    global vocab_src, vocab_tgt, spacy_de, spacy_en

    # Calculate safe number of processes based on available memory
    available_memory = psutil.virtual_memory().available
    safe_parallel_count = min(number_of_parallelized_experiments, max(1, int(available_memory / (2 * 1024 * 1024 * 1024))))
    
    if safe_parallel_count < number_of_parallelized_experiments:
        print(f"Reducing parallel experiments from {number_of_parallelized_experiments} to {safe_parallel_count}")
        number_of_parallelized_experiments = safe_parallel_count

    # Create fresh separated directory structure
    if os.path.exists("./separated"):
       
        shutil.rmtree("./separated")
    os.makedirs("./separated")

    # Build auxiliary graphs first
    build_auxiliary_graphs(inject_parameters)

    # Create process directories and copy files
    for i in range(number_of_parallelized_experiments):
        process_dir = f"./separated/process_{i}"
        os.makedirs(process_dir, exist_ok=True)
        for f in ["tensor_to_inject.onnx", "layer_to_inject.onnx", "rest_of_layers.onnx"]:
            src = os.path.join("./separated", f)
            if os.path.exists(src):
                dst = os.path.join(process_dir, f)
                shutil.copy2(src, dst)

    _, valid_dataloader = create_dataloaders(
        torch.device("cpu"),
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=1,
        is_distributed=False,
    )

    outer_batch = []
    for b_outer_index in range(number_of_parallelized_experiments):
        batch = []
        for b_inner_index in range(n_examples):
            batch.append(next(iter(valid_dataloader)))
        outer_batch.append(batch)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(12345 + np.random.randint(0, 1000))
    
    try:
        mp.spawn(
            worker,
            args=(outer_batch, model_path, vocab_src, vocab_tgt, inject_parameters, n_examples),
            nprocs=number_of_parallelized_experiments,
            join=True
        )
    finally:
        # Clean up
        del outer_batch
        if os.path.exists("./separated"):
            shutil.rmtree("./separated")

    return None

def worker(gpu_id, batches, model_path, vocab_src, vocab_tgt, inject_parameters, n_examples):
    try:
        # Add gpu_id to inject_parameters for process-specific paths
        inject_parameters = copy.deepcopy(inject_parameters)
        inject_parameters['gpu_id'] = gpu_id
        
        batch = batches[gpu_id]
        return check_outputs(model_path, vocab_src, vocab_tgt, inject_parameters, 
                           n_examples, 2, "</s>", batch)
    except Exception as e:
        print(f"Worker {gpu_id} failed with error: {str(e)}")
        import traceback
        traceback.print_exc()  # Print full stack trace
        raise e

def get_weight_dict(module_path):
    module = ModelWrapper(module_path)
    module_graph = module.graph
    module_weights = module.graph.initializer
    module_weight_dict = {}
    for weight in module_weights:
        module_weight_dict[weight.name] = numpy_helper.to_array(weight)
    return module_graph, module_weight_dict

def prepare_inference(module_path, module_input_values, inject_parameters=None):
    # If using separated directory, route to process-specific directory
    if inject_parameters and 'gpu_id' in inject_parameters:
        if './separated/' in module_path:
            process_dir = f"./separated/process_{inject_parameters['gpu_id']}"
            module_path = os.path.join(process_dir, os.path.basename(module_path))
            print(f"Using process-specific path: {module_path}")

    if inject_parameters:
        try:
            layer_model = onnx.load(module_path)
            for value_info in layer_model.graph.value_info:
                if (value_info.name) in list(module_input_values.keys()):
                    layer_model.graph.value_info.remove(value_info)
                if (value_info.name) == inject_parameters["faulty_output_tensor"]:
                    layer_model.graph.value_info.remove(value_info)
            onnx.save(layer_model, module_path)
        except Exception as e:
            print(f"Error processing ONNX file {module_path}: {str(e)}")
            raise

    module = ModelWrapper(module_path)
    output = [node.name for node in module.graph.output]

    input_all = [node.name for node in module.graph.input]
    input_initializers = [node.name for node in module.graph.initializer]
    module_input_names = list(set(input_all) - set(input_initializers))

    module_graph, module_weight_dict = get_weight_dict(module_path)

    for input_name in module_input_names:
        module_weight_dict[input_name] = module_input_values[input_name]

    return module_weight_dict, module_graph

# def greedy_decode(model, src, src_mask, max_len, start_symbol, custom_decoder=False, inject_parameters=None):
#     # Helper function to get process-specific path
#     def get_process_path(base_path):
#         if inject_parameters and 'gpu_id' in inject_parameters and './separated/' in base_path:
#             process_dir = f"./separated/process_{inject_parameters['gpu_id']}"
#             return os.path.join(process_dir, os.path.basename(base_path))
#         return base_path
#     src_float = model.get_src_embed(src)

#     if exists("weights/encoder.pt"):
#         encoder_weight_dict, encoder_graph = torch.load("weights/encoder.pt")
#     else:
#         encoder_weight_dict, encoder_graph = prepare_inference("./try/encoder_try_cleaned.onnx", {"global_in": src_float.detach().numpy(), "global_in_1": src_mask.detach().numpy()})
#         torch.save((encoder_weight_dict, encoder_graph), "./weights/encoder.pt")
#     ort_sess_encoder = ort.InferenceSession('./try/encoder_try_cleaned.onnx')

#     if (not inject_parameters) or ("Encoder" not in inject_parameters["targetted_module"]):
#         memory = torch.from_numpy(ort_sess_encoder.run(None, {
#             "global_in": src_float.detach().numpy(), 
#             "global_in_1": src_mask.detach().numpy()
#         })[0])
#     else:
#         if "RANDOM" in inject_parameters["inject_type"]:
#             ort_pre_inject_encoder = ort.InferenceSession('./separated/layer_to_inject.onnx')
#             tensor_to_inject = ort_pre_inject_encoder.run(None, {
#                 "global_in": src_float.detach().numpy(), 
#                 "global_in_1": src_mask.detach().numpy(),
#             })[0]

#             target_indices = [np.random.randint(0, dim) for dim in tensor_to_inject.shape]
#             golden_value = tensor_to_inject[tuple(target_indices)]
#             if "BITFLIP" in inject_parameters["inject_type"]:
#                 print("RANDOM BITFLIP FAULTY:")
#                 faulty_value, flip_bit = float32_bit_flip(tensor_to_inject, target_indices)
#             else:
#                 print("RANDOM FAULTY:")
#                 faulty_value = delta_init()
#             print(tensor_to_inject[tuple(target_indices)])
#             tensor_to_inject[tuple(target_indices)] = faulty_value
#             print(tensor_to_inject[tuple(target_indices)])
#             ort_sess_rest_of_encoder = ort.InferenceSession("./separated/rest_of_layers.onnx")
#             memory = ort_sess_rest_of_encoder.run(None, {
#                 "global_in": src_float.detach().numpy(), 
#                 "global_in_1": src_mask.detach().numpy(),
#                 inject_parameters["faulty_output_tensor"]: tensor_to_inject
#             })[0]
#         else:
#             ort_pre_inject_encoder = ort.InferenceSession('./separated/tensor_to_inject.onnx')
#             tensor_to_inject = torch.from_numpy(ort_pre_inject_encoder.run(None, {
#                 "global_in": src_float.detach().numpy(), 
#                 "global_in_1": src_mask.detach().numpy(),
#             })[0])
#             layer_to_inject_weight_dict, layer_to_inject_encoder_graph = prepare_inference("./separated/layer_to_inject.onnx", {
#                 "global_in": src_float.detach().numpy(), 
#                 "global_in_1": src_mask.detach().numpy(),
#                 inject_parameters["faulty_tensor_name"]: tensor_to_inject.detach().numpy(),
#             }, inject_parameters)
#             faulty_layer_output, _ = run_module("Encoder", {
#                 "global_in": src_float.detach().numpy(), 
#                 "global_in_1": src_mask.detach().numpy(),
#                 inject_parameters["faulty_tensor_name"]: tensor_to_inject.detach().numpy(),
#             }, "./separated/layer_to_inject.onnx", layer_to_inject_weight_dict, layer_to_inject_encoder_graph, inject_parameters)
#             #TODO: Fix "None" appearing in faulty_layer_output
#             print("SEE HERE:")
#             print(tensor_to_inject)
#             print(faulty_layer_output)
#             faulty_layer_output = torch.from_numpy(faulty_layer_output[list(faulty_layer_output.keys())[0]])
#             """
#             print(np.nonzero(faulty_layer_output))
#             print(faulty_layer_output.dtype)
#             """
#             ort_sess_rest_of_encoder = ort.InferenceSession("./separated/rest_of_layers.onnx")
#             memory = ort_sess_rest_of_encoder.run(None, {
#                 "global_in": src_float.detach().numpy(), 
#                 "global_in_1": src_mask.detach().numpy(),
#                 inject_parameters["faulty_output_tensor"]: faulty_layer_output.detach().numpy(),
#             })[0]
#         memory_2 = ort_sess_encoder.run(None, {
#             "global_in": src_float.detach().numpy(), 
#             "global_in_1": src_mask.detach().numpy()
#         })[0]

#         print("DIFFERENCES:")
#         print(np.sum(memory != memory_2))
#         print("SIMILARITIES:")
#         print(np.sum(memory == memory_2))

#         memory = torch.from_numpy(memory)

#     ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)

#     ys_float = model.get_tgt_embed(ys)

#     if exists("weights/decoder.pt"):
#         decoder_weight_dict, decoder_graph = torch.load("weights/decoder.pt")
#     else:
#         decoder_weight_dict, decoder_graph = prepare_inference("./try/decoder_try_cleaned.onnx", {
#                 "global_in": ys_float.detach().numpy(),
#                 "global_in_1": memory.detach().numpy(),
#                 "global_in_2": src_mask.detach().numpy(),
#                 "global_in_3": subsequent_mask(ys.size(1)).type_as(src.data).detach().numpy(),
#         })
#         torch.save((decoder_weight_dict, decoder_graph), "./weights/decoder.pt")

#     ort_sess_decoder = ort.InferenceSession("./try/decoder_try_cleaned.onnx")
#     for i in range(max_len - 1):
#         #print(str(i) + "/" + str(max_len-1))
#         ys_float = model.get_tgt_embed(ys)
        
#         pass_inject_parameters = None
#         custom_decoder = False

#         if inject_parameters and (i == (inject_parameters["target_inference_number"] - 1)) and ("Decoder" in inject_parameters["targetted_module"]):
#             pass_inject_parameters = inject_parameters
#             custom_decoder = True

#         start_time = time.time()
#         if custom_decoder:
#             if "RANDOM" in inject_parameters["inject_type"]:
#                 ort_pre_inject_decoder = ort.InferenceSession('./separated/layer_to_inject.onnx')
#                 tensor_to_inject = ort_pre_inject_decoder.run(None, {
#                     "global_in": ys_float.detach().numpy(),
#                     "global_in_1": memory.detach().numpy(),
#                     "global_in_2": src_mask.detach().numpy(),
#                     "global_in_3": subsequent_mask(ys.size(1)).type_as(src.data).detach().numpy(),
#                 })[0]
#                 target_indices = [np.random.randint(0, dim) for dim in tensor_to_inject.shape]
#                 golden_value = tensor_to_inject[tuple(target_indices)]
#                 if "BITFLIP" in inject_parameters["inject_type"]:
#                     print("RANDOM BITFLIP FAULTY:")
#                     faulty_value, flip_bit = float32_bit_flip(tensor_to_inject, target_indices)
#                 else:
#                     print("RANDOM FAULTY:")
#                     faulty_value = delta_init()
#                 print(tensor_to_inject[tuple(target_indices)])
#                 tensor_to_inject[tuple(target_indices)] = faulty_value
#                 print(tensor_to_inject[tuple(target_indices)])
#                 ort_sess_rest_of_decoder = ort.InferenceSession("./separated/rest_of_layers.onnx")
#                 out = (torch.from_numpy(ort_sess_rest_of_decoder.run(None, {
#                     "global_in": ys_float.detach().numpy(),
#                     "global_in_1": memory.detach().numpy(),
#                     "global_in_2": src_mask.detach().numpy(),
#                     "global_in_3": subsequent_mask(ys.size(1)).type_as(src.data).detach().numpy(),
#                     inject_parameters["faulty_output_tensor"]: tensor_to_inject,
#                 })[0]))
#             else:
#                 ort_pre_inject_decoder = ort.InferenceSession('./separated/tensor_to_inject.onnx')
#                 tensor_to_inject = ort_pre_inject_decoder.run(None, {
#                     "global_in": ys_float.detach().numpy(),
#                     "global_in_1": memory.detach().numpy(),
#                     "global_in_2": src_mask.detach().numpy(),
#                     "global_in_3": subsequent_mask(ys.size(1)).type_as(src.data).detach().numpy(),
#                 })[0]
#                 tensor_to_inject = torch.from_numpy(tensor_to_inject)
#                 _, layer_to_inject_decoder_graph = prepare_inference("./separated/layer_to_inject.onnx", {
#                         "global_in": ys_float.detach().numpy(),
#                         "global_in_1": memory.detach().numpy(),
#                         "global_in_2": src_mask.detach().numpy(),
#                         "global_in_3": subsequent_mask(ys.size(1)).type_as(src.data).detach().numpy(),
#                         inject_parameters["faulty_tensor_name"]: tensor_to_inject.detach().numpy(),
#                 }, pass_inject_parameters)
#                 faulty_layer_output, _ = run_module("Decoder", {
#                     "global_in": ys_float.detach().numpy(),
#                     "global_in_1": memory.detach().numpy(),
#                     "global_in_2": src_mask.detach().numpy(),
#                     "global_in_3": subsequent_mask(ys.size(1)).type_as(src.data).detach().numpy(),
#                     inject_parameters["faulty_tensor_name"]: tensor_to_inject.detach().numpy(),
#                 }, "./separated/layer_to_inject.onnx", decoder_weight_dict, layer_to_inject_decoder_graph, pass_inject_parameters)
#                 faulty_layer_output = torch.from_numpy(faulty_layer_output[list(faulty_layer_output.keys())[0]])
#                 #print(faulty_layer_output)
#                 ort_sess_rest_of_decoder = ort.InferenceSession("./separated/rest_of_layers.onnx")
#                 out = (torch.from_numpy(ort_sess_rest_of_decoder.run(None, {
#                     "global_in": ys_float.detach().numpy(),
#                     "global_in_1": memory.detach().numpy(),
#                     "global_in_2": src_mask.detach().numpy(),
#                     "global_in_3": subsequent_mask(ys.size(1)).type_as(src.data).detach().numpy(),
#                     inject_parameters["faulty_output_tensor"]: faulty_layer_output.detach().numpy(),
#                 })[0]))

#             out_2 = (torch.from_numpy(ort_sess_decoder.run(None, {
#                 "global_in": ys_float.detach().numpy(),
#                 "global_in_1": memory.detach().numpy(),
#                 "global_in_2": src_mask.detach().numpy(),
#                 "global_in_3": subsequent_mask(ys.size(1)).type_as(src.data).detach().numpy(),
#             })[0]))

#             prob = model.generator(out_2[:, -1])
#             _, next_word = torch.max(prob, dim=1)
#             golden_next_word = next_word.data[0]
#             associated_values, next_words = torch.topk(prob, 2, dim=1)

#             print("GOLDEN NEXT_WORD:")
#             print(next_words, associated_values)
#             print(golden_next_word)

#             del next_word, prob

#             prob = model.generator(out[:, -1])
#             _, next_word = torch.max(prob, dim=1)
#             next_word = next_word.data[0]
#             associated_values, next_words = torch.topk(prob, 2, dim=1)

#             print("FAULTY NEXT_WORD:")
#             print(next_words, associated_values)
#             print(next_word)
#             print(next_word==golden_next_word)

#             if (next_word!=golden_next_word):
#                 print("TOKEN CHANGED!")

#             ys = torch.cat(
#                 [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
#             )
#             continue
#         else:
#             out = torch.from_numpy(ort_sess_decoder.run(None, {
#                 "global_in": ys_float.detach().numpy(),
#                 "global_in_1": memory.detach().numpy(),
#                 "global_in_2": src_mask.detach().numpy(),
#                 "global_in_3": subsequent_mask(ys.size(1)).type_as(src.data).detach().numpy(),
#             })[0])
#         prob = model.generator(out[:, -1])
#         _, next_word = torch.max(prob, dim=1)
#         next_word = next_word.data[0]
#         ys = torch.cat(
#             [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
#         )
#     return ys

def greedy_decode(model, src, src_mask, max_len, start_symbol, custom_decoder=False, inject_parameters=None):
    # Helper function to get process-specific path
    def get_process_path(base_path):
        if inject_parameters and 'gpu_id' in inject_parameters and './separated/' in base_path:
            process_dir = f"./separated/process_{inject_parameters['gpu_id']}"
            return os.path.join(process_dir, os.path.basename(base_path))
        return base_path
    src_float = model.get_src_embed(src)

    if exists("weights/encoder.pt"):
        encoder_weight_dict, encoder_graph = torch.load("weights/encoder.pt")
    else:
        encoder_weight_dict, encoder_graph = prepare_inference("./try/encoder_try_cleaned.onnx", {"global_in": src_float.detach().numpy(), "global_in_1": src_mask.detach().numpy()})
        torch.save((encoder_weight_dict, encoder_graph), "./weights/encoder.pt")
    ort_sess_encoder = ort.InferenceSession('./try/encoder_try_cleaned.onnx')

    if (not inject_parameters) or ("Encoder" not in inject_parameters["targetted_module"]):
        memory = torch.from_numpy(ort_sess_encoder.run(None, {
            "global_in": src_float.detach().numpy(), 
            "global_in_1": src_mask.detach().numpy()
        })[0])
    else:
        if "RANDOM" in inject_parameters["inject_type"]:
            # Use process-specific path
            process_layer_path = get_process_path('./separated/layer_to_inject.onnx')
            ort_pre_inject_encoder = ort.InferenceSession(process_layer_path)
            tensor_to_inject = ort_pre_inject_encoder.run(None, {
                "global_in": src_float.detach().numpy(), 
                "global_in_1": src_mask.detach().numpy(),
            })[0]

            target_indices = [np.random.randint(0, dim) for dim in tensor_to_inject.shape]
            golden_value = tensor_to_inject[tuple(target_indices)]
            if "BITFLIP" in inject_parameters["inject_type"]:
                print("RANDOM BITFLIP FAULTY:")
                faulty_value, flip_bit = float32_bit_flip(tensor_to_inject, target_indices)
            else:
                print("RANDOM FAULTY:")
                faulty_value = delta_init()
            print(tensor_to_inject[tuple(target_indices)])
            tensor_to_inject[tuple(target_indices)] = faulty_value
            print(tensor_to_inject[tuple(target_indices)])
            process_rest_path = get_process_path('./separated/rest_of_layers.onnx')
            ort_sess_rest_of_encoder = ort.InferenceSession(process_rest_path)
            memory = ort_sess_rest_of_encoder.run(None, {
                "global_in": src_float.detach().numpy(), 
                "global_in_1": src_mask.detach().numpy(),
                inject_parameters["faulty_output_tensor"]: tensor_to_inject
            })[0]
        else:
            # Use process-specific paths
            process_tensor_path = get_process_path('./separated/tensor_to_inject.onnx')
            process_layer_path = get_process_path('./separated/layer_to_inject.onnx')
            process_rest_path = get_process_path('./separated/rest_of_layers.onnx')
            ort_pre_inject_encoder = ort.InferenceSession(process_tensor_path)
            tensor_to_inject = torch.from_numpy(ort_pre_inject_encoder.run(None, {
                "global_in": src_float.detach().numpy(), 
                "global_in_1": src_mask.detach().numpy(),
            })[0])

            # prepare_inference already handles process paths
            layer_to_inject_weight_dict, layer_to_inject_encoder_graph = prepare_inference(
                process_layer_path,  # Now using process path
                {
                    "global_in": src_float.detach().numpy(), 
                    "global_in_1": src_mask.detach().numpy(),
                    inject_parameters["faulty_tensor_name"]: tensor_to_inject.detach().numpy(),
                }, 
                inject_parameters
            )

            faulty_layer_output, _ = run_module("Encoder", {
                "global_in": src_float.detach().numpy(), 
                "global_in_1": src_mask.detach().numpy(),
                inject_parameters["faulty_tensor_name"]: tensor_to_inject.detach().numpy(),
            }, process_layer_path,  # Now using process path
               layer_to_inject_weight_dict, 
               layer_to_inject_encoder_graph, 
               inject_parameters)

            faulty_layer_output = torch.from_numpy(faulty_layer_output[list(faulty_layer_output.keys())[0]])
            
            ort_sess_rest_of_encoder = ort.InferenceSession(process_rest_path)  # Now using process path
            memory = ort_sess_rest_of_encoder.run(None, {
                "global_in": src_float.detach().numpy(), 
                "global_in_1": src_mask.detach().numpy(),
                inject_parameters["faulty_output_tensor"]: faulty_layer_output.detach().numpy(),
            })[0]
        memory_2 = ort_sess_encoder.run(None, {
            "global_in": src_float.detach().numpy(), 
            "global_in_1": src_mask.detach().numpy()
        })[0]

        print("DIFFERENCES:")
        print(np.sum(memory != memory_2))
        print("SIMILARITIES:")
        print(np.sum(memory == memory_2))

        memory = torch.from_numpy(memory)

    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)

    ys_float = model.get_tgt_embed(ys)

    if exists("weights/decoder.pt"):
        decoder_weight_dict, decoder_graph = torch.load("weights/decoder.pt")
    else:
        decoder_weight_dict, decoder_graph = prepare_inference("./try/decoder_try_cleaned.onnx", {
                "global_in": ys_float.detach().numpy(),
                "global_in_1": memory.detach().numpy(),
                "global_in_2": src_mask.detach().numpy(),
                "global_in_3": subsequent_mask(ys.size(1)).type_as(src.data).detach().numpy(),
        })
        torch.save((decoder_weight_dict, decoder_graph), "./weights/decoder.pt")

    ort_sess_decoder = ort.InferenceSession("./try/decoder_try_cleaned.onnx")
    for i in range(max_len - 1):
        #print(str(i) + "/" + str(max_len-1))
        ys_float = model.get_tgt_embed(ys)
        
        pass_inject_parameters = None
        custom_decoder = False

        if inject_parameters and (i == (inject_parameters["target_inference_number"] - 1)) and ("Decoder" in inject_parameters["targetted_module"]):
            pass_inject_parameters = inject_parameters
            custom_decoder = True

        start_time = time.time()
        if custom_decoder:
            if "RANDOM" in inject_parameters["inject_type"]:
                process_layer_path = get_process_path('./separated/layer_to_inject.onnx')
                ort_pre_inject_decoder = ort.InferenceSession(process_layer_path)
                tensor_to_inject = ort_pre_inject_decoder.run(None, {
                    "global_in": ys_float.detach().numpy(),
                    "global_in_1": memory.detach().numpy(),
                    "global_in_2": src_mask.detach().numpy(),
                    "global_in_3": subsequent_mask(ys.size(1)).type_as(src.data).detach().numpy(),
                })[0]
                target_indices = [np.random.randint(0, dim) for dim in tensor_to_inject.shape]
                golden_value = tensor_to_inject[tuple(target_indices)]
                if "BITFLIP" in inject_parameters["inject_type"]:
                    print("RANDOM BITFLIP FAULTY:")
                    faulty_value, flip_bit = float32_bit_flip(tensor_to_inject, target_indices)
                else:
                    print("RANDOM FAULTY:")
                    faulty_value = delta_init()
                print(tensor_to_inject[tuple(target_indices)])
                tensor_to_inject[tuple(target_indices)] = faulty_value
                print(tensor_to_inject[tuple(target_indices)])
                process_rest_path = get_process_path('./separated/rest_of_layers.onnx')
                ort_sess_rest_of_decoder = ort.InferenceSession(process_rest_path)
                out = (torch.from_numpy(ort_sess_rest_of_decoder.run(None, {
                    "global_in": ys_float.detach().numpy(),
                    "global_in_1": memory.detach().numpy(),
                    "global_in_2": src_mask.detach().numpy(),
                    "global_in_3": subsequent_mask(ys.size(1)).type_as(src.data).detach().numpy(),
                    inject_parameters["faulty_output_tensor"]: tensor_to_inject,
                })[0]))
            else:
                process_tensor_path = get_process_path('./separated/tensor_to_inject.onnx')
                process_layer_path = get_process_path('./separated/layer_to_inject.onnx')
                process_rest_path = get_process_path('./separated/rest_of_layers.onnx')
                ort_pre_inject_decoder = ort.InferenceSession(process_tensor_path)
                tensor_to_inject = ort_pre_inject_decoder.run(None, {
                    "global_in": ys_float.detach().numpy(),
                    "global_in_1": memory.detach().numpy(),
                    "global_in_2": src_mask.detach().numpy(),
                    "global_in_3": subsequent_mask(ys.size(1)).type_as(src.data).detach().numpy(),
                })[0]
                tensor_to_inject = torch.from_numpy(tensor_to_inject)
                _, layer_to_inject_decoder_graph = prepare_inference(process_layer_path, {
                        "global_in": ys_float.detach().numpy(),
                        "global_in_1": memory.detach().numpy(),
                        "global_in_2": src_mask.detach().numpy(),
                        "global_in_3": subsequent_mask(ys.size(1)).type_as(src.data).detach().numpy(),
                        inject_parameters["faulty_tensor_name"]: tensor_to_inject.detach().numpy(),
                }, pass_inject_parameters)
                faulty_layer_output, _ = run_module("Decoder", {
                    "global_in": ys_float.detach().numpy(),
                    "global_in_1": memory.detach().numpy(),
                    "global_in_2": src_mask.detach().numpy(),
                    "global_in_3": subsequent_mask(ys.size(1)).type_as(src.data).detach().numpy(),
                    inject_parameters["faulty_tensor_name"]: tensor_to_inject.detach().numpy(),
                }, process_layer_path, decoder_weight_dict, layer_to_inject_decoder_graph, pass_inject_parameters)
                faulty_layer_output = torch.from_numpy(faulty_layer_output[list(faulty_layer_output.keys())[0]])
                #print(faulty_layer_output)
                ort_sess_rest_of_decoder = ort.InferenceSession(process_rest_path)
                out = (torch.from_numpy(ort_sess_rest_of_decoder.run(None, {
                    "global_in": ys_float.detach().numpy(),
                    "global_in_1": memory.detach().numpy(),
                    "global_in_2": src_mask.detach().numpy(),
                    "global_in_3": subsequent_mask(ys.size(1)).type_as(src.data).detach().numpy(),
                    inject_parameters["faulty_output_tensor"]: faulty_layer_output.detach().numpy(),
                })[0]))

            out_2 = (torch.from_numpy(ort_sess_decoder.run(None, {
                "global_in": ys_float.detach().numpy(),
                "global_in_1": memory.detach().numpy(),
                "global_in_2": src_mask.detach().numpy(),
                "global_in_3": subsequent_mask(ys.size(1)).type_as(src.data).detach().numpy(),
            })[0]))

            prob = model.generator(out_2[:, -1])
            _, next_word = torch.max(prob, dim=1)
            golden_next_word = next_word.data[0]
            associated_values, next_words = torch.topk(prob, 2, dim=1)

            print("GOLDEN NEXT_WORD:")
            print(next_words, associated_values)
            print(golden_next_word)

            del next_word, prob

            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.data[0]
            associated_values, next_words = torch.topk(prob, 2, dim=1)

            print("FAULTY NEXT_WORD:")
            print(next_words, associated_values)
            print(next_word)
            print(next_word==golden_next_word)

            if (next_word!=golden_next_word):
                print("TOKEN CHANGED!")

            ys = torch.cat(
                [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
            )
            continue
        else:
            out = torch.from_numpy(ort_sess_decoder.run(None, {
                "global_in": ys_float.detach().numpy(),
                "global_in_1": memory.detach().numpy(),
                "global_in_2": src_mask.detach().numpy(),
                "global_in_3": subsequent_mask(ys.size(1)).type_as(src.data).detach().numpy(),
            })[0])
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys



def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0


def load_trained_model():
    model_path = "checkpoint/iwslt14_model_final.pt"

    module = str(args.module)
    directory_name = str(args.directory_name)
    directory_list = os.listdir(directory_name)
    bit_width = 8

    if module == "Encoder":
        weight_dict, main_graph = torch.load("weights/encoder.pt")
    else:
        weight_dict, main_graph = torch.load("weights/decoder.pt")

    for layer in directory_list:
        #for fault_model in ["WEIGHT16"]:#["INPUT", "WEIGHT", "INPUT16", "WEIGHT16", "RANDOM", "RANDOM_BITFLIP"]:
        for fault_model in ["INPUT", "WEIGHT", "INPUT16", "WEIGHT16", "RANDOM", "RANDOM_BITFLIP"]:
            for bit_position in range(8):
                input_inject_data = json.load(open(directory_name + "/" + layer))
                faulty_bit_position = None
                if "RANDOM" not in fault_model:
                    faulty_bit_position = int(bit_position)
                """
                print(input_inject_data)
                """
                (input_quantizer_name, int_input_tensor_name), (weight_quantizer_name, int_weight_tensor_name), _, (input_trace, weight_trace) = get_target_inputs(main_graph, input_inject_data["target_layer"], input_inject_data["input_tensor"], input_inject_data["weight_tensor"], None, input_inject_data["output_tensor"])
                original_weight_dict = weight_dict.copy()
                faulty_quantizer_name = None
                faulty_tensor_name = None
                if "INPUT" in fault_model:
                    faulty_trace = input_trace
                    faulty_quantizer_name = input_quantizer_name
                    faulty_tensor_name = int_input_tensor_name
                elif "WEIGHT" in fault_model:
                    faulty_trace = weight_trace
                    faulty_quantizer_name = weight_quantizer_name
                    faulty_tensor_name = int_weight_tensor_name
                elif "RANDOM" in fault_model:
                    faulty_trace = None
                    faulty_quantizer = None
                    faulty_tensor_name = input_inject_data["output_tensor"]

                # Target first generated token (target_inference_number)
                # Inject i = target_inference_number, where i is the i-th token for inference
                # For now just inject the first inference location
                total_experiments = 8
                number_of_parallelized_experiments = 2
                target_inference_number = 1

                assert ((total_experiments % number_of_parallelized_experiments) == 0)
                total_experiments = total_experiments // number_of_parallelized_experiments

                inject_parameters = {}
                inject_parameters["inject_type"] = fault_model
                inject_parameters["faulty_tensor_name"] = faulty_tensor_name 
                inject_parameters["faulty_quantizer_name"] = faulty_quantizer_name
                inject_parameters["faulty_trace"] = faulty_trace
                inject_parameters["faulty_bit_position"] = faulty_bit_position
                inject_parameters["faulty_output_tensor"] = input_inject_data["output_tensor"]
                inject_parameters["faulty_operation_name"] = input_inject_data["target_layer"]
                inject_parameters["targetted_module"] = input_inject_data["module"] 
                inject_parameters["target_inference_number"] = target_inference_number
                inject_parameters["experiment_output_file"] = str(args.experiment_output_name)

                print("FAULT MODEL:")
                print(fault_model, faulty_bit_position)
                run_model_example(model_path, inject_parameters, total_experiments, number_of_parallelized_experiments)
                break
        exit()

if is_interactive_notebook():
    model = load_trained_model()