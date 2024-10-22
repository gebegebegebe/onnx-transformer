import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant.scaled_int import Int32Bias
from brevitas.quant.scaled_int import Int8ActPerTensorFloat
from brevitas.quant.scaled_int import Uint8ActPerTensorFloat
from brevitas.quant.scaled_int import Int8WeightPerChannelFloat

bit_width = 8

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        """
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        """
        self.w_1 = qnn.QuantLinear(d_model, d_ff, bit_width=bit_width, weight_bit_width=bit_width, weight_quant=Int8WeightPerChannelFloat, bias=True, bias_quant=Int32Bias, input_quant=Uint8ActPerTensorFloat, return_quant_tensor=True)
        self.w_2 = qnn.QuantLinear(d_ff, d_model, bit_width=bit_width, weight_bit_width=bit_width, weight_quant=Int8WeightPerChannelFloat, bias=True, bias_quant=Int32Bias, input_quant=Uint8ActPerTensorFloat, return_quant_tensor=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = nn.functional.relu(x)
        return self.w_2(self.dropout(x))
