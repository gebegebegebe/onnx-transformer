import os
from os.path import exists
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets
import GPUtil
import warnings
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from model import make_model
from label_smoothing import LabelSmoothing
from batch import Batch
import torchtext
import nltk
import onnxruntime as ort
ort.set_default_logger_severity(3)
import numpy as np
from inject_operations import run_module
import copy

warnings.filterwarnings("ignore")
RUN_EXAMPLES = True

import argparse
parser = argparse.ArgumentParser('Fault Injection Program')
parser.add_argument('--directory_name')
args = parser.parse_args()
import json

from inject_utils.layers import *
from inject_utils.utils import *
from qonnx.core.modelwrapper import ModelWrapper


def tokenize(text):
    temp = text.split(" ")
    return temp


def load_vocab():
    vocab_src, vocab_tgt = torch.load("vocab.pt")
    print("Finished.\nVocabulary sizes:")
    print(len(vocab_src))
    print(len(vocab_tgt))
    return vocab_src, vocab_tgt


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
    batch_size=12000,
    max_padding=128,
    is_distributed=True,
):
    # def create_dataloaders(batch_size=12000):
    def tokenize_de(text):
        return tokenize(text)

    def tokenize_en(text):
        return tokenize(text)

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

def clean_text_output(string_list):
    if "</s>" not in string_list:
        return None
    string_list = (" ".join(string_list))
    string_list = string_list.replace("@@ ", "")
    string_list = string_list.replace("<s> ", "")
    string_list = string_list.replace(" &apos;", "'")
    string_list = string_list.split(" ")
    end_index = (string_list.index("</s>"))
    string_list = string_list[:end_index]
    return string_list

def check_outputs(
    valid_dataloader,
    model,
    vocab_src,
    vocab_tgt,
    inject_parameters,
    n_examples=15,
    pad_idx=2,
    eos_string="</s>",
):
    n_examples = 1
    results = [()] * n_examples
    reference_text = []
    output_text = []
    model.eval()
    #for idx in range(n_examples):
    for idx in range(n_examples):
        print("\nExample %d ========\n" % idx)
        b = next(iter(valid_dataloader))
        rb = Batch(b[0], b[1], pad_idx)
        greedy_decode(model, rb.src, rb.src_mask, 64, 0, inject_parameters, False, False)[0]

        src_tokens = [
            vocab_src.get_itos()[x] for x in rb.src[0] if x != pad_idx
        ]
        tgt_tokens = [
            vocab_tgt.get_itos()[x] for x in rb.tgt[0] if x != pad_idx
        ]

        print(
            "Source Text (Input)        : "
            + " ".join(src_tokens).replace("\n", "")
        )
        print(
            "Target Text (Ground Truth) : "
            + " ".join(tgt_tokens).replace("\n", "")
        )
        model_out = greedy_decode(model, rb.src, rb.src_mask, 72, 0, inject_parameters, False, False)[0]
        model_txt = (
            " ".join(
                [vocab_tgt.get_itos()[x] for x in model_out if x != pad_idx]
            ).split(eos_string, 1)[0]
            + eos_string
        )

        # Target Text

        output_target = clean_text_output(tgt_tokens)
        if output_target is None:
            continue
        reference_text.append([output_target])

        # Model Output Text (Fault-Free)

        output_list = [vocab_tgt.get_itos()[x] for x in model_out if x != pad_idx]
        output_list = clean_text_output(output_list)
        if output_list is None:
            continue
        output_text.append(output_list)

        #TODO Model Output Text (Faulty)

        model_out_faulty = greedy_decode(model, rb.src, rb.src_mask, 72, 0, inject_parameters, False, True)[0]
        model_txt_faulty = (
            " ".join(
                [vocab_tgt.get_itos()[x] for x in model_out_faulty if x != pad_idx]
            ).split(eos_string, 1)[0]
            + eos_string
        )

        output_list_faulty = [vocab_tgt.get_itos()[x] for x in model_out_faulty if x != pad_idx]
        output_list_faulty = clean_text_output(output_list_faulty)

        #TODO fix this base case:
        if output_list_faulty is None:
            continue

        # Print out everything else

        print("TARGET:")
        print(output_target)
        print("MODEL OUTPUT:")
        print(output_list)
        print("MODEL OUTPUT FAULTY:")
        print(output_list_faulty)
        print("SENTENCE BLEU:")
        print(nltk.translate.bleu_score.sentence_bleu([output_target], output_list))
        results[idx] = (rb, src_tokens, tgt_tokens, model_out, model_txt)
    print("CORPUS BLEU:")
    bleu_score = (nltk.translate.bleu_score.corpus_bleu(reference_text, output_text))
    print(bleu_score)
    return bleu_score 


def run_model_example(model_path, inject_parameters, n_examples=5):
    vocab_src, vocab_tgt = load_vocab()

    print("Preparing Data ...")
    _, valid_dataloader = create_dataloaders(
        torch.device("cpu"),
        vocab_src,
        vocab_tgt,
        batch_size=1,
        is_distributed=False,
    )

    print("Loading Trained Model ...")

    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device("cpu"))
    )

    print("Checking Model Outputs:")
    example_data = check_outputs(
        valid_dataloader, model, vocab_src, vocab_tgt, inject_parameters, n_examples=n_examples
    )
    return model, example_data


def greedy_decode(model, src, src_mask, max_len, start_symbol, inject_parameters, custom_decoder=False, inject_fault=False):
    src_float = model.get_src_embed(src)
    encoder_weight_dict, encoder_graph = torch.load("weights/encoder.pt")
    memory, _ = run_module("encoder", {"global_in": src_float.detach().numpy(), "global_in_1": src_mask.detach().numpy()}, "./onnx/new_fixed/encoder_fixed.onnx", encoder_weight_dict, encoder_graph, inject_parameters)
    memory = torch.from_numpy(memory[list(memory.keys())[0]])
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)

    import time
    decoder_weight_dict, decoder_graph = torch.load("weights/decoder.pt")

    for i in range(max_len - 1):
        print(str(i) + "/" + str(max_len-1))
        ys_float = model.get_tgt_embed(ys)
        start_time = time.time()

        input_data = {
            "global_in": ys_float.detach().numpy(),
            "global_in_1": memory.detach().numpy(),
            "global_in_2": src_mask.detach().numpy(),
            "global_in_3": subsequent_mask(ys.size(1)).type_as(src.data).detach().numpy(),
        }
        if custom_decoder:
            current_decoder_graph = copy.deepcopy(decoder_graph)
            out, _ = run_module("decoder", input_data, "./onnx/new_fixed/decoder_fixed.onnx", decoder_weight_dict, current_decoder_graph, inject_paramaters)
            out = torch.from_numpy(out[list(out.keys())[0]])
            del current_decoder_graph
        else:
            ort_sess_decoder = ort.InferenceSession('./onnx/new_fixed/decoder_fixed.onnx')
            out = torch.from_numpy(ort_sess_decoder.run(None, input_data)[0])
        print(time.time() - start_time)

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
    model_path = "checkpoint/iwslt14_model_00.pt"

    directory_name = str(args.directory_name)
    directory_list = os.listdir(directory_name)
    bit_width = 4

    weight_dict, main_graph = torch.load("weights/encoder.pt")

    for layer in directory_list:
        for fault_model in ["INPUT", "WEIGHT", "INPUT16", "WEIGHT16", "RANDOM", "RANDOM_BITFLIP"]:
            for bit_position in range(4):
                input_inject_data = json.load(open(directory_name + "/" + layer))
                faulty_bit_position = None
                if "RANDOM" not in fault_model:
                    faulty_bit_position = int(bit_position)
                (input_quantizer_name, int_input_tensor_name), (weight_quantizer_name, int_weight_tensor_name), _ = get_target_inputs(main_graph, input_inject_data["target_layer"], input_inject_data["input_tensor"], input_inject_data["weight_tensor"], None, input_inject_data["output_tensor"])
                original_weight_dict = weight_dict.copy()

                faulty_quantizer_name = None
                faulty_tensor_name = None 
                if "INPUT" in fault_model:
                    faulty_quantizer_name = input_quantizer_name
                    faulty_tensor_name = int_input_tensor_name
                elif "WEIGHT" in fault_model:
                    faulty_quantizer_name = weight_quantizer_name
                    faulty_tensor_name = int_weight_tensor_name
                elif "RANDOM" in fault_model:
                    faulty_quantizer_name = None
                    faulty_tensor_name = output_tensor_name

                #Injection parameters:
                inject_parameters = {}
                inject_parameters["original_weight_dict"] = original_weight_dict
                inject_parameters["main_graph"] = copy.deepcopy(main_graph)
                inject_parameters["inject_type"] = fault_model
                inject_parameters["faulty_tensor_name"] = faulty_tensor_name 
                inject_parameters["faulty_quantizer_name"] = faulty_quantizer_name 
                inject_parameters["faulty_bit_position"] = faulty_bit_position
                inject_parameters["faulty_operation_name"] = input_inject_data["target_layer"]
                run_model_example(model_path, inject_parameters, 1)

                #print(inject_parameters)
                exit()

                

        #run_model_example(model_path, input_inject_data)


if __name__ == "__main__":
    model = load_trained_model()
