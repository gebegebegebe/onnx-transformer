from onnx import helper, ModelProto, TensorProto, OperatorSetIdProto, shape_inference
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
import onnx.numpy_helper as numpy_helper
import numpy as np
import torch
from inject_utils.utils import *
import inject_utils.layers
import time

def execute_node(node, main_graph, final_output_node, weight_dict, module, inject_input):
    """
    if ("Transpose" in node.name):
        print(node)

    if (inject_input and False):
        # Disable debug temporarily with False
        debug_inject_parameters(inject_input)
    """

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

    if inject_input:
        if ("RANDOM" not in inject_input["inject_type"]) and (node.op_type == "DequantizeLinear") and (inject_input["faulty_quantizer_name"] in node.name):
            weight_dict, dequantized_operation_input_name, faulty_indices, golden_bit_value, faulty_bit_value, is_signed = inject_utils.layers.perturb_quantizer(model, input_dict, weight_dict, inject_input["faulty_tensor_name"], inject_input["faulty_bit_position"])
            inject_input["dequantized_operation_input_name"] = dequantized_operation_input_name

        if ("RANDOM" in inject_input["inject_type"]):
            if (inject_input) and (inject_input["faulty_operation_name"] in node.name):
                faulty_value = None
                target_indices = [np.random.randint(0, dim) for dim in weight_dict[inject_input["faulty_tensor_name"]].shape]
                golden_value = weight_dict[inject_input["faulty_tensor_name"]][tuple(target_indices)]
                if "BITFLIP" in inject_input["inject_type"]:
                    faulty_value, float32_bit_position = inject_utils.layers.float32_bit_flip(weight_dict[inject_input["faulty_tensor_name"]], target_indices)
                else:
                    faulty_value = delta_init(True)
                weight_dict[inject_input["faulty_tensor_name"]][tuple(target_indices)] = faulty_value
                faulty_indices = target_indices

        if "INPUT" in inject_input["inject_type"] or "WEIGHT" in inject_input["inject_type"]:
            if (node.op_type == "MatMul") and (node.name == inject_input["faulty_operation_name"]):
                if not (inject_input["dequantized_operation_input_name"]):
                    print("Error with dequantized value")
                    sys.exit(0)

                """
                transpose_dimensions = None
                for individual_node in node_inputs:
                    if "Transpose" in individual_node.name:
                        transpose_dimensions = []
                        for dimension in (individual_node.type.tensor_type.shape.dim):
                            transpose_dimensions.append(int(dimension.dim_value))

                print(transpose_dimensions)

                print("NODE INPUT:")
                print(node_inputs)
                print(dir(node_inputs))
                print(node_inputs[1].name)
                print(node_inputs[1].type.tensor_type.shape.dim[1])
                """

                delta_perturb = inject_utils.layers.perturb_matmul(model, input_dict, weight_dict, inject_input["dequantized_operation_input_name"], inject_input["transposed_axes"])
                print("HERE:")
                print(np.nonzero(delta_perturb))

                perturb_result = np.add(original_tensor_output,delta_perturb)
                output_tensors[tensor_output_name] = perturb_result
                weight_dict[tensor_output_name] = perturb_result
                print("DONE")
                #exit()

    return output_tensors, weight_dict, list_operation_time, inject_input

def inference(main_graph, weight_dict, module, inject_input):
    def execute_single_node(node, weight_dict, main_graph, module, inject_input):
        final_output_node = node.output[0]
        output_tensors, weight_dict, list_operation_time, inject_input = execute_node(node, main_graph, final_output_node, weight_dict, module, inject_input)
        return output_tensors, weight_dict, list_operation_time, inject_input
    output_tensors = None
    for node in main_graph.node:
        output_tensors, weight_dict, list_operation_time, inject_input = execute_single_node(node, weight_dict, main_graph, module, inject_input)
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

def run_module(module, input_values, module_filepath, module_weight_dict, module_graph, inject_input=None):
    """
    if inject_input:
        print("WILL INJECT!")
        print(module)
        for key in inject_input.keys():
            if key != "main_graph" and key != "original_weight_dict":
                print("--")
                print(key)
                print(inject_input[key])
                print("--")
        return
    """
    for input_name in list(input_values.keys()):
        module_weight_dict[input_name] = input_values[input_name]

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
