from onnx import helper, ModelProto, TensorProto, OperatorSetIdProto, shape_inference
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
import onnx.numpy_helper as numpy_helper
import numpy as np

def execute_node(node, main_graph, final_output_node, weight_dict):
    node_inputs = []
    node_outputs = []

    added_quant_inputs, added_quant_outputs = expand_node_inputs_outputs(main_graph, node)
    node_inputs += added_quant_inputs
    node_outputs += added_quant_outputs

    desired_node_outputs = [x for x in node_outputs if x.name == final_output_node]
    intermediate_node_outputs = [x for x in node_outputs if x.name != final_output_node]

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
    if node.input[0] in fix_dictionary.keys():
        new_tensor = helper.ValueInfoProto()
        new_tensor.CopyFrom(node_inputs[0])
        new_tensor.name = fix_dictionary[node.input[0]]
        node_inputs.append(new_tensor)
        weight_dict[new_tensor.name] = weight_dict[node_inputs[0].name]

    graph = helper.make_graph(
            nodes = [node],
            name = "single_node_exec",
            inputs = node_inputs,
            outputs = desired_node_outputs
    )

    model = ModelProto()
    model.graph.CopyFrom(graph)
    model.opset_import.append(OperatorSetIdProto(version=13))
    model = ModelWrapper(model)

    input_dict = {}
    for node_iter in node_inputs:
        if node_iter.name == [node_intermediate.name for node_intermediate in intermediate_node_outputs]:
            continue
        if node_iter.name in [node_intermediate.name for node_intermediate in node_outputs]:
            continue
        input_dict[node_iter.name] = weight_dict[node_iter.name]

    output_tensors = execute_onnx(model, input_dict)
    tensor_output_name = list(output_tensors.keys())[0]
    original_tensor_output = output_tensors[tensor_output_name]
    weight_dict[tensor_output_name] = output_tensors[tensor_output_name]

    return output_tensors, weight_dict

def inference(main_graph, weight_dict):
    def execute_single_node(node, weight_dict, main_graph):
        final_output_node = node.output[0]
        output_tensors, weight_dict = execute_node(node, main_graph, final_output_node, weight_dict)
        return output_tensors, weight_dict
    output_tensors = None
    for node in main_graph.node:
        output_tensors, weight_dict = execute_single_node(node, weight_dict, main_graph)
    return output_tensors, weight_dict

def expand_node_inputs_outputs(graph, node):
    added_inputs = []
    added_outputs = []

    added_inputs += list(filter(lambda x: x.name in node.input, graph.input))
    added_inputs += list(filter(lambda x: x.name in node.input, graph.output))
    added_inputs += list(filter(lambda x: x.name in node.input, graph.value_info))
    added_outputs += list(filter(lambda x: x.name in node.output, graph.output))
    added_outputs += list(filter(lambda x: x.name in node.output, graph.value_info))

    return added_inputs, added_outputs

def get_weight_dict(module_path):
    module = ModelWrapper(module_path)
    module_graph = module.graph
    module_weights = module.graph.initializer
    module_weight_dict = {}
    for weight in module_weights:
        module_weight_dict[weight.name] = numpy_helper.to_array(weight)
    return module, module_graph, module_weight_dict


if __name__ == "__main__":
    input_1_name = "global_in"
    input_2_name = "global_in_1"
    encoder_filename = "./onnx/encoder.onnx"
    decoder_filename = "./onnx/decoder.onnx"
    encoder, encoder_graph, encoder_weight_dict = get_weight_dict(encoder_filename)

    tensor_float32 = np.random.rand(1, 128, 512).astype(np.float32)
    tensor_boolean = np.random.choice([True, False], size=(1, 1, 128))

    encoder_weight_dict[input_1_name] = tensor_float32
    encoder_weight_dict[input_2_name] = tensor_boolean
    output_tensors, weight_dict = inference(encoder_graph, encoder_weight_dict)
    print(output_tensors)
