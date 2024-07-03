import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant.scaled_int import Int32Bias

bit_width = 4

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        #self.proj = nn.Linear(d_model, vocab)
        self.proj = qnn.QuantLinear(d_model, vocab, bit_width=bit_width, weight_bit_width=bit_width, bias=True, quant_bias=Int32Bias)

    def forward(self, x):
        return nn.functional.log_softmax(self.proj(x), dim=-1)
