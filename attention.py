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
    def __init__(self, h, d_model, dropout=0.1, num_tokens_1=72, num_tokens_2=72):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        #self.linears = clones(qnn.QuantLinear(d_model, d_model, bit_width=bit_width, weight_bit_width=bit_width, weight_quant=Int8WeightPerChannelFloat, bias=True, bias_quant=Int32Bias, input_bit_width=bit_width), 4)

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

        self.weights_query = nn.Parameter(torch.randn(d_model, d_model))
        self.weights_key = nn.Parameter(torch.randn(d_model, d_model))
        self.weights_value = nn.Parameter(torch.randn(d_model, d_model))
        self.weights_final = nn.Parameter(torch.randn(d_model, d_model))
        self.quantizer_weights_query = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True, scaling_per_output_channel=True, scaling_stats_permute_dims=(1, 0), per_channel_broadcastable_shape=(1, d_model))
        self.quantizer_weights_key = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True, scaling_per_output_channel=True, scaling_stats_permute_dims=(1, 0), per_channel_broadcastable_shape=(1, d_model))
        self.quantizer_weights_value = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True, scaling_per_output_channel=True, scaling_stats_permute_dims=(1, 0), per_channel_broadcastable_shape=(1, d_model))
        self.quantizer_weights_output = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True, scaling_per_output_channel=True, scaling_stats_permute_dims=(1, 0), per_channel_broadcastable_shape=(1, d_model))

        """
        self.quantizer_1 = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True, scaling_per_output_channel=True, scaling_stats_permute_dims=(1, 2, 0, 3), per_channel_broadcastable_shape=(1, 1, num_tokens, 1))
        self.quantizer_2 = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True, scaling_per_output_channel=True, scaling_stats_permute_dims=(1, 2, 0, 3), per_channel_broadcastable_shape=(1, 1, 72, 1))
        self.quantizer_3 = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True, scaling_per_output_channel=True, scaling_stats_permute_dims=(1, 2, 0, 3), per_channel_broadcastable_shape=(1, 1, 72, 1))
        self.quantizer_4 = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True, scaling_per_output_channel=True, scaling_stats_permute_dims=(1, 2, 0, 3), per_channel_broadcastable_shape=(1, 1, num_tokens, 1))
        """

        self.quantizer_query = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True, scaling_per_output_channel=True, scaling_stats_permute_dims=(1, 0, 2), per_channel_broadcastable_shape=(1, num_tokens_1, 1))
        self.quantizer_key = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True, scaling_per_output_channel=True, scaling_stats_permute_dims=(1, 0, 2), per_channel_broadcastable_shape=(1, num_tokens_2, 1))
        self.quantizer_value = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True, scaling_per_output_channel=True, scaling_stats_permute_dims=(1, 0, 2), per_channel_broadcastable_shape=(1, num_tokens_2, 1))
        self.quantizer_output = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True, scaling_per_output_channel=True, scaling_stats_permute_dims=(1, 0, 2), per_channel_broadcastable_shape=(1, num_tokens_1, 1))

        #self.quantizers = [self.quantizer_1, self.quantizer_2, self.quantizer_3, self.quantizer_4]


    def attention(self, query, key, value, mask=None, dropout=None, quantizers=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)

        #scores = torch.matmul(self.quantizers[0](query), self.quantizers[1](key.transpose(-2, -1))) \
                 #/ math.sqrt(d_k)
        scores = torch.matmul((query), (key.transpose(-2, -1))) \
                 / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = nn.functional.softmax(scores, dim = -1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        """
        print("--")
        print(query.shape)
        print(key.shape)
        print(value.shape)
        print(p_attn.shape)
        print("--")
        """
        #return torch.matmul(self.quantizers[2](p_attn), self.quantizers[3](value)), p_attn
        return torch.matmul((p_attn), (value)), p_attn

        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        print(self.quantizer_query(query).shape, self.quantizer_key(key).shape, self.quantizer_value(value).shape)
        #[l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        query, key, value = \
            [torch.matmul(x, w).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             #for l, x, w in zip(self.linears, (self.quantizer_query(query), self.quantizer_key(key), self.quantizer_value(value)), (self.weights_query, self.weights_key, self.weights_value))]
             for x, w in zip((self.quantizer_query(query), self.quantizer_key(key), self.quantizer_value(value)), (self.quantizer_weights_query(self.weights_query), self.quantizer_weights_key(self.weights_key), self.quantizer_weights_value(self.weights_value)))]
        print(query.shape, key.shape, value.shape)
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = self.attention(query, key, value, mask, self.dropout, None)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return torch.matmul(self.quantizer_output(x), self.weights_final)
