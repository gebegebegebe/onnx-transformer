import torch
import torch.nn as nn

import functools
from collections import defaultdict

from functools import partial
import numpy as np
from tqdm import tqdm

import os
from torchtext.data.functional import to_map_style_dataset
from model import make_model
from torch.utils.data import DataLoader
from torch.nn.functional import log_softmax, pad
from batch import Batch


def tokenize(text):
    temp = text.split(" ")
    return temp

def tokenize_de(text):
    return tokenize(text)

def tokenize_en(text):
    return tokenize(text)


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

#def get_act_scales(model, tokenizer, dataset_path, num_samples=512, seq_len=512):
def get_act_scales(model, dataset, vocab_tgt, num_samples=512, seq_len=128):
    model.eval()
    device = next(model.parameters()).device
    act_scales = {}
    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(functools.partial(stat_input_hook, name=name))
            )

    """
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.shuffle(seed=42)

    for i in tqdm(range(num_samples)):
        input_ids = tokenizer(
            dataset[i]["text"], return_tensors="pt", max_length=seq_len, truncation=True
        ).input_ids#.to(device)
        model(input_ids)
    """
    counter = 0
    for batch in tqdm(dataset):
        if counter > num_samples:
            break
        counter = counter + 1
        #for batch in Batch(batch[0], batch[1], pad_idx):
        model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)

    for h in hooks:
        h.remove()

    return act_scales

def main():
    def collate_fn(batch):
        device="cpu"
        max_padding=128
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

    vocab_src, vocab_tgt = load_vocab()
    model = make_model(len(vocab_src), len(vocab_tgt), N=6)

    train_iter = create_dataset("./data/train.de.bpe","./data/train.en.bpe")
    valid_iter = create_dataset("./data/valid.de.bpe","./data/valid.en.bpe")
    test_iter = create_dataset("./data/test.de.bpe","./data/test.en.bpe")

    train_iter_map = to_map_style_dataset(
        train_iter
    )  # DistributedSampler needs a dataset len()
    train_sampler = (
        None
    )
    valid_iter_map = to_map_style_dataset(valid_iter)
    valid_sampler = (
        None
    )
    train_dataloader = DataLoader(
        train_iter_map,
        batch_size=1,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_iter_map,
        batch_size=1,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler,
        collate_fn=collate_fn,
    )

    pad_idx = vocab_tgt["<blank>"]
    act_scales = get_act_scales(model, (Batch(b[0], b[1], pad_idx) for b in valid_dataloader), vocab_tgt, 10, 128)
    torch.save(act_scales, "scales/transformer_scales.pt")
    print(torch.load("scales/transformer_scales.pt").keys())

if __name__ == "__main__":
    main()
