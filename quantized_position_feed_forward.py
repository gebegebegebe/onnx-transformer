import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant.scaled_int import Int32Bias
from brevitas.quant.scaled_int import Int8ActPerTensorFloat
from brevitas.quant.scaled_int import Uint8ActPerTensorFloat
from brevitas.quant.scaled_int import Int8WeightPerChannelFloat

bit_width = 8

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1, num_tokens=72):
        super(PositionwiseFeedForward, self).__init__()
        """
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        """
        """
        self.w_1 = qnn.QuantLinear(d_model, d_ff, bit_width=bit_width, weight_bit_width=bit_width, weight_quant=Int8WeightPerChannelFloat, bias=True, bias_quant=Int32Bias, input_quant=Uint8ActPerTensorFloat, return_quant_tensor=True)
        self.w_2 = qnn.QuantLinear(d_ff, d_model, bit_width=bit_width, weight_bit_width=bit_width, weight_quant=Int8WeightPerChannelFloat, bias=True, bias_quant=Int32Bias, input_quant=Uint8ActPerTensorFloat, return_quant_tensor=True)
        """
        self.quantizer_1 = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True, scaling_per_output_channel=True, scaling_stats_permute_dims=(1, 0, 2), per_channel_broadcastable_shape=(1, num_tokens, 1))
        self.quantizer_2 = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True, scaling_per_output_channel=True, scaling_stats_permute_dims=(1, 0), per_channel_broadcastable_shape=(1, d_ff))
        self.quantizer_3 = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True, scaling_per_output_channel=True, scaling_stats_permute_dims=(1, 0, 2), per_channel_broadcastable_shape=(1, num_tokens, 1))
        self.quantizer_4 = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True, scaling_per_output_channel=True, scaling_stats_permute_dims=(1, 0), per_channel_broadcastable_shape=(1, d_model))

        self.weights_1 = nn.Parameter(torch.randn(d_model, d_ff))
        self.weights_2 = nn.Parameter(torch.randn(d_ff, d_model))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        print("LINEAR:")
        print(x.shape)
        print(self.weights_1.shape)
        print(self.weights_2.shape)
        print("LINEAR:")
        x = torch.matmul(self.quantizer_1(x), self.quantizer_2(self.weights_1))
        #x = self.w_1(x)
        x = nn.functional.relu(x)
        #return self.w_2(self.dropout(x))
        return torch.matmul(self.dropout(self.quantizer_3(x)), self.quantizer_4(self.weights_2))
