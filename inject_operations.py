from onnx import helper, ModelProto, TensorProto, OperatorSetIdProto, shape_inference
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
import onnx.numpy_helper as numpy_helper
import numpy as np
import torch

import time

def execute_node(node, main_graph, final_output_node, weight_dict, module, inject_input):
    print(inject_input)
    exit()
    node_inputs = []
    node_outputs = []

    added_quant_inputs, added_quant_outputs, list_operation_time = expand_node_inputs_outputs(main_graph, node, weight_dict, module)
    node_inputs += added_quant_inputs
    node_outputs += added_quant_outputs

    desired_node_outputs = [x for x in node_outputs if x.name == final_output_node]
    intermediate_node_outputs = [x for x in node_outputs if x.name != final_output_node]
    
    graph = helper.make_graph(
            nodes = [node],
            name = "single_node_exec",
            inputs = node_inputs,
            outputs = desired_node_outputs
    )

    model = ModelProto()
    model = shape_inference.infer_shapes(model)
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

    return output_tensors, weight_dict, list_operation_time

def inference(main_graph, weight_dict, module, inject_input):
    def execute_single_node(node, weight_dict, main_graph, module, inject_input):
        final_output_node = node.output[0]
        output_tensors, weight_dict, list_operation_time = execute_node(node, main_graph, final_output_node, weight_dict, module, inject_input)
        return output_tensors, weight_dict, list_operation_time
    output_tensors = None
    for node in main_graph.node:
        output_tensors, weight_dict, list_operation_time = execute_single_node(node, weight_dict, main_graph, module, inject_input)
    return output_tensors, weight_dict

def expand_node_inputs_outputs(graph, node, weight_dict, module):
    added_inputs = []
    added_outputs = []

    added_inputs += list(filter(lambda x: x.name in node.input, graph.input))
    added_inputs += list(filter(lambda x: x.name in node.input, graph.output))
    added_inputs += list(filter(lambda x: x.name in node.input, graph.value_info))
    added_outputs += list(filter(lambda x: x.name in node.output, graph.output))
    added_outputs += list(filter(lambda x: x.name in node.output, graph.value_info))

    start_time = time.time()
    if module == "decoder":
        replacement_dictionary = {
            "onnx::ReduceMean_0_dynamic_axes_1": weight_dict["global_in"].shape[1],
            "onnx::Unsqueeze_3_dynamic_axes_1": weight_dict["global_in_3"].shape[1],
            "onnx::Unsqueeze_3_dynamic_axes_2": weight_dict["global_in_3"].shape[2],
        }

        for input_tensor in added_inputs:
            for dimension in range(len(input_tensor.type.tensor_type.shape.dim)):
                for key in replacement_dictionary.keys():
                    if key in str(input_tensor.type.tensor_type.shape.dim[dimension]):
                        input_tensor.type.tensor_type.shape.dim[dimension].Clear()
                        input_tensor.type.tensor_type.shape.dim[dimension].dim_value = replacement_dictionary[key]
                    if "unk__" in str(input_tensor.type.tensor_type.shape.dim[dimension]):
                        input_tensor.type.tensor_type.shape.dim[dimension].Clear()
                        input_tensor.type.tensor_type.shape.dim[dimension].dim_value = weight_dict[input_tensor.name].shape[dimension]

    return added_inputs, added_outputs, time.time() - start_time

def get_weight_dict(module_path):
    module = ModelWrapper(module_path)
    module_graph = module.graph
    module_weights = module.graph.initializer
    module_weight_dict = {}
    for weight in module_weights:
        module_weight_dict[weight.name] = numpy_helper.to_array(weight)
    return module_graph, module_weight_dict

def prepare_inference(module_path, module_input_values):
    module = ModelWrapper(module_path)
    output = [node.name for node in module.graph.output]

    input_all = [node.name for node in module.graph.input]
    input_initializers = [node.name for node in module.graph.initializer]
    module_input_names = list(set(input_all) - set(input_initializers))

    module_graph, module_weight_dict = get_weight_dict(module_path)

    for input_name in module_input_names:
        module_weight_dict[input_name] = module_input_values[input_name]

    return module_weight_dict, module_graph

def run_module(module, input_values, module_filepath, module_weight_dict, module_graph, inject_input):
    start_time = time.time()
    for input_name in list(input_values.keys()):
        module_weight_dict[input_name] = input_values[input_name]
    print("LOAD TIME: " + str(time.time() - start_time))

    return inference(module_graph, module_weight_dict, module, inject_input)

if __name__ == "__main__":

    module = "encoder"
    encoder_input_values = {
        "global_in": np.random.rand(1, 128, 512).astype(np.float32), 
        "global_in_1": np.random.choice([True, False], size=(1, 1, 128))}
    module_filepath = "./onnx/new_fixed/encoder_fixed.onnx"
    output_tensors, module_weight_dict = run_module(module, encoder_input_values, module_filepath)
    torch.save(module_weight_dict, "encoder.pt")

    print("ENCODER OUT:")
    print(output_tensors)

    module = "decoder"
    decoder_input_values = {
        "global_in": np.random.rand(1, 1, 512).astype(np.float32), 
        "global_in_1": np.random.rand(1, 128, 512).astype(np.float32), 
        "global_in_2": np.random.choice([True, False], size=(1, 1, 128)),
        "global_in_3": np.random.rand(1, 1, 1).astype(np.int64)}
    module_filepath = "./onnx/new_fixed/decoder_fixed.onnx"

    output_tensors, module_weight_dict = run_module(module, decoder_input_values, module_filepath)
    torch.save(module_weight_dict, "decoder.pt")

    print("DECODER OUT:")
    print(output_tensors)
