from onnx import helper, ModelProto, TensorProto, OperatorSetIdProto, shape_inference
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
import onnx.numpy_helper as numpy_helper
import numpy as np

def execute_node(node, main_graph, final_output_node, weight_dict, module):
    node_inputs = []
    node_outputs = []

    added_quant_inputs, added_quant_outputs = expand_node_inputs_outputs(main_graph, node, weight_dict)
    node_inputs += added_quant_inputs
    node_outputs += added_quant_outputs

    desired_node_outputs = [x for x in node_outputs if x.name == final_output_node]
    intermediate_node_outputs = [x for x in node_outputs if x.name != final_output_node]
    
    #module = "encoder"
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

    return output_tensors, weight_dict

def inference(main_graph, weight_dict, module):
    def execute_single_node(node, weight_dict, main_graph, module):
        final_output_node = node.output[0]
        output_tensors, weight_dict = execute_node(node, main_graph, final_output_node, weight_dict, module)
        return output_tensors, weight_dict
    output_tensors = None
    for node in main_graph.node:
        output_tensors, weight_dict = execute_single_node(node, weight_dict, main_graph, module)
    return output_tensors, weight_dict

def expand_node_inputs_outputs(graph, node, weight_dict):
    added_inputs = []
    added_outputs = []

    added_inputs += list(filter(lambda x: x.name in node.input, graph.input))
    added_inputs += list(filter(lambda x: x.name in node.input, graph.output))
    added_inputs += list(filter(lambda x: x.name in node.input, graph.value_info))
    added_outputs += list(filter(lambda x: x.name in node.output, graph.output))
    added_outputs += list(filter(lambda x: x.name in node.output, graph.value_info))

    replacement_dictionary = {
        "onnx::ReduceMean_0_dynamic_axes_1": 1,
        "onnx::Unsqueeze_3_dynamic_axes_1": 1,
        "onnx::Unsqueeze_3_dynamic_axes_2": 1,
    }

    for input_tensor in added_inputs:
        for dimension in range(len(input_tensor.type.tensor_type.shape.dim)):
            #print(input_tensor.type.tensor_type.shape.dim[dimension])
            for key in replacement_dictionary.keys():
                if key in str(input_tensor.type.tensor_type.shape.dim[dimension]):
                    #print("FOUND")
                    input_tensor.type.tensor_type.shape.dim[dimension].Clear()
                    input_tensor.type.tensor_type.shape.dim[dimension].dim_value = replacement_dictionary[key]
                if "unk__" in str(input_tensor.type.tensor_type.shape.dim[dimension]):
                    #print(weight_dict[input_tensor.name].shape[dimension])
                    input_tensor.type.tensor_type.shape.dim[dimension].Clear()
                    input_tensor.type.tensor_type.shape.dim[dimension].dim_value = weight_dict[input_tensor.name].shape[dimension]
                    #print(input_tensor)

    return added_inputs, added_outputs

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

if __name__ == "__main__":

    module = "encoder"
    encoder_input_values = {
        "global_in": np.random.rand(1, 128, 512).astype(np.float32), 
        "global_in_1": np.random.choice([True, False], size=(1, 1, 128))}
    module_weight_dict, module_graph = prepare_inference("./onnx/encoder.onnx", encoder_input_values)
    output_tensors, module_weight_dict = inference(module_graph, module_weight_dict, module)

    print("ENCODER OUT:")
    print(output_tensors)

    module = "decoder"
    decoder_input_values = {
        "global_in": np.random.rand(1, 1, 512).astype(np.float32), 
        "global_in_1": np.random.rand(1, 128, 512).astype(np.float32), 
        "global_in_2": np.random.choice([True, False], size=(1, 1, 128)),
        "global_in_3": np.random.rand(1, 1, 1).astype(np.int64)}
    module_weight_dict, module_graph = prepare_inference("./onnx/decoder.onnx", decoder_input_values)
    output_tensors, module_weight_dict = inference(module_graph, module_weight_dict, module)

    print("DECODER OUT:")
    print(output_tensors)
