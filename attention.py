import torch
import torch.nn as nn

import math
from utils import clones

import brevitas.nn as qnn
from brevitas.quant.scaled_int import Int32Bias
from brevitas.quant.scaled_int import Uint8ActPerTensorFloat
from brevitas.quant.scaled_int import Int8WeightPerChannelFloat

bit_width = 8

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, channels=128):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(qnn.QuantLinear(d_model, d_model, bit_width=bit_width, weight_bit_width=bit_width, bias=True, bias_quant=Int32Bias, input_quant=Uint8ActPerTensorFloat, input_bit_width=bit_width), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.quantizer_1 = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True, scaling_per_output_channel=True, scaling_stats_permute_dims=(2, 1, 0, 3), per_channel_broadcastable_shape=(1, 1, channels, 1))
        self.quantizer_2 = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True, scaling_per_output_channel=True, scaling_stats_permute_dims=(3, 1, 2, 0), per_channel_broadcastable_shape=(1, 1, 1, channels))
        self.quantizer_3 = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True, scaling_per_output_channel=True, scaling_stats_permute_dims=(3, 1, 2, 0), per_channel_broadcastable_shape=(1, 1, 1, channels))
        self.quantizer_4 = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True, scaling_per_output_channel=True, scaling_stats_permute_dims=(2, 1, 0, 3), per_channel_broadcastable_shape=(1, 1, channels, 1))
        self.quantizers = [self.quantizer_1, self.quantizer_2, self.quantizer_3, self.quantizer_4]
        self.channels = channels


    def attention(self, query, key, value, mask=None, dropout=None, quantizers=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)

        _, _, channels, _ = (value.shape)
        print("--")
        print("value:")
        print(value.shape)
        print("key:")
        print(key.shape)
        print("query:")
        print(query.shape)

        # TODO: Fix for decoder
        scores = torch.matmul(self.quantizers[0](query), self.quantizers[1](key.transpose(-2, -1))) \
                 / math.sqrt(d_k)
        """
        scores = torch.matmul((query), (key.transpose(-2, -1))) \
                 / math.sqrt(d_k)
        """
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = nn.functional.softmax(scores, dim = -1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        print("scores:")
        print(scores.shape)
        print("p_attn:")
        print(p_attn.shape)
        print("channels:")
        print(self.channels)
        print("--")
        return torch.matmul(self.quantizers[2](p_attn), self.quantizers[3](value)), p_attn
        #return torch.matmul((p_attn), (value)), p_attn

        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = self.attention(query, key, value, mask, self.dropout, None)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        #return self.linears[-1](self.quantizer_2(x))
        return self.linears[-1](x)
