import onnx
import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
import onnx.numpy_helper as numpy_helper
import torch
from brevitas.export import export_onnx_qcdq
import qonnx.util.cleanup


# Load the ONNX model
model_filename = "decoder_try.onnx"
model = ModelWrapper(model_filename)
model = qonnx.util.cleanup.cleanup(in_file=model_filename, out_file=model_filename)
model = ModelWrapper(model_filename)

#model_weight_dict = torch.load("./weight_dict/decoder_weight_dict.pt")

"""
module = "decoder"
if module == "encoder":
    fix_dictionary = {
        "Sub_0_out0": "/layers.0/sublayer.0/norm/Sub_output_0",
        "Sub_2_out0": "/layers.0/sublayer.1/norm/Sub_output_0",
        "Sub_4_out0": "/layers.1/sublayer.0/norm/Sub_output_0",
        "Sub_6_out0": "/layers.1/sublayer.1/norm/Sub_output_0",
        "Sub_8_out0": "/layers.2/sublayer.0/norm/Sub_output_0",
        "Sub_10_out0": "/layers.2/sublayer.1/norm/Sub_output_0",
        "Sub_12_out0": "/layers.3/sublayer.0/norm/Sub_output_0",
        "Sub_14_out0": "/layers.3/sublayer.1/norm/Sub_output_0",
        "Sub_16_out0": "/layers.4/sublayer.0/norm/Sub_output_0",
        "Sub_18_out0": "/layers.4/sublayer.1/norm/Sub_output_0",
        "Sub_20_out0": "/layers.5/sublayer.0/norm/Sub_output_0",
        "Sub_22_out0": "/layers.5/sublayer.1/norm/Sub_output_0",
        "Sub_24_out0": "/norm/Sub_output_0"
    }
else:
    fix_dictionary = {
        "Sub_0_out0": "/layers.0/sublayer.0/norm/Sub_output_0",
        "Sub_2_out0": "/layers.0/sublayer.1/norm/Sub_output_0",
        "Sub_4_out0": "/layers.0/sublayer.2/norm/Sub_output_0",
        "Sub_6_out0": "/layers.1/sublayer.0/norm/Sub_output_0",
        "Sub_8_out0": "/layers.1/sublayer.1/norm/Sub_output_0",
        "Sub_10_out0": "/layers.1/sublayer.2/norm/Sub_output_0",
        "Sub_12_out0": "/layers.2/sublayer.0/norm/Sub_output_0",
        "Sub_14_out0": "/layers.2/sublayer.1/norm/Sub_output_0",
        "Sub_16_out0": "/layers.2/sublayer.2/norm/Sub_output_0",
        "Sub_18_out0": "/layers.3/sublayer.0/norm/Sub_output_0",
        "Sub_20_out0": "/layers.3/sublayer.1/norm/Sub_output_0",
        "Sub_22_out0": "/layers.3/sublayer.2/norm/Sub_output_0",
        "Sub_24_out0": "/layers.4/sublayer.0/norm/Sub_output_0",
        "Sub_26_out0": "/layers.4/sublayer.1/norm/Sub_output_0",
        "Sub_28_out0": "/layers.4/sublayer.2/norm/Sub_output_0",
        "Sub_30_out0": "/layers.5/sublayer.0/norm/Sub_output_0",
        "Sub_32_out0": "/layers.5/sublayer.1/norm/Sub_output_0",
        "Sub_34_out0": "/layers.5/sublayer.2/norm/Sub_output_0",
        "Sub_36_out0": "/norm/Sub_output_0"
    }

fix_dictionary = {v: k for k, v in fix_dictionary.items()}
for node in model.graph.node:
    print("---")
    for i in range(len(node.input)):
        if node.input[i] in list(fix_dictionary.keys()):
            node.input[i] = fix_dictionary[node.input[i]]
    print(node.input)
    print("---")
"""

"""
for key, value in fix_dictionary.items():
    new_initializer = onnx.helper.make_tensor(
        name=value,
        data_type=onnx.TensorProto.FLOAT,
        dims=model_weight_dict[key].shape,
        vals=model_weight_dict[key]
    )
    model.graph.initializer.append(new_initializer)

for value_info in model.graph.value_info:
    print(value_info)
"""

# Save the modified model
#model.save("test/decoder_fixed.onnx")
