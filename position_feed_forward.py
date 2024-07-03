import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant.scaled_int import Int32Bias
from brevitas.quant.scaled_int import Int8ActPerTensorFloat

bit_width = 4

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        """
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        """
        self.w_1 = qnn.QuantLinear(d_model, d_ff, bit_width=bit_width, weight_bit_width=bit_width, bias=True, quant_bias=Int32Bias)
        self.w_2 = qnn.QuantLinear(d_ff, d_model, bit_width=bit_width, weight_bit_width=bit_width, bias=True, quant_bias=Int32Bias)
        self.dropout = nn.Dropout(dropout)
        self.quantizer_1 = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=False)
        self.quantizer_2 = qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=False)

    def forward(self, x):
        x = self.quantizer_1(x)
        #return self.w_2(self.dropout(nn.functional.relu(self.w_1(x))))
        return self.w_2(self.dropout(self.quantizer_2(self.w_1(x))))
