# ONNX-transformer

Quantized transformer (8-bit and 4-bit) exported to ONNX

- For ONNX export use _output.py_
- For inference please check _./reference_

Additional notes:

- Directly works for 8-bit
- For 4-bit use the Brevitas QuantLinear layers then train (through QAT)
