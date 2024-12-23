import torch.nn as nn
import copy

from attention import MultiHeadedAttention
from encoder_decoder import EncoderDecoder
from position_feed_forward import PositionwiseFeedForward
from positional_encodings import PositionalEncoding
from embeddings import Embeddings
from encoder import *
from decoder import *
from generator import Generator

import brevitas.nn as qnn

def make_model(src_vocab, tgt_vocab, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    dropout = 0.3
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)

    attn_2 = MultiHeadedAttention(h, d_model, num_tokens_1=71, num_tokens_2=71)
    attn_3 = MultiHeadedAttention(h, d_model, num_tokens_1=71, num_tokens_2=72)

    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    ff_2 = PositionwiseFeedForward(d_model, d_ff, dropout, num_tokens=71)

    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, attn_2, attn_3,
                             ff_2, dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model
