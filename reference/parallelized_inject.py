from finn.core.modelwrapper import ModelWrapper
from finn.core.onnx_exec import execute_onnx
from onnx import helper, ModelProto, TensorProto, OperatorSetIdProto, shape_inference
from layers import *
from utils.utils import *
from tensorflow.keras.datasets import cifar10
from multiprocessing import Pool
from functools import partial
import onnxruntime as rt
import onnx.numpy_helper as numpy_helper
import numpy as np
import copy
import time
import json
import sys
import os

import argparse
parser = argparse.ArgumentParser('Fault Injection Program')
parser.add_argument('--directory_name')
parser.add_argument('--bit_width')
args = parser.parse_args()

def inference(main_graph, weight_dict, inject_fault, inject_type, faulty_tensor_name, faulty_quantizer_name, faulty_bit_position, weight_tensor_name, bias_output_name, faulty_indices, faulty_operation_name):
    def execute_single_node(node, weight_dict, main_graph, dequantized_operation_input_name, faulty_indices, float32_bit_position, golden_bit_value, faulty_bit_value, is_signed):
        final_output_node = node.output[0]
        output_tensors, weight_dict, dequantized_operation_input_name, faulty_indices, float32_bit_position, golden_bit_value, faulty_bit_value, is_signed = execute_node(node, main_graph, final_output_node, weight_dict, inject_fault, inject_type, faulty_tensor_name, faulty_quantizer_name, faulty_bit_position, dequantized_operation_input_name, weight_tensor_name, bias_output_name, faulty_indices, faulty_operation_name, float32_bit_position, golden_bit_value, faulty_bit_value, is_signed)
        return output_tensors, weight_dict, dequantized_operation_input_name, faulty_indices, float32_bit_position, golden_bit_value, faulty_bit_value, is_signed
    output_tensors = None
    dequantized_operation_input_name = None
    faulty_indices = None
    float32_bit_position = None
    golden_bit_value = None
    faulty_bit_value = None
    is_signed = None
    for node in main_graph.node:
        output_tensors, weight_dict, dequantized_operation_input_name, faulty_indices, float32_bit_position, golden_bit_value, faulty_bit_value, is_signed = execute_single_node(node, weight_dict, main_graph, dequantized_operation_input_name, faulty_indices, float32_bit_position, golden_bit_value, faulty_bit_value, is_signed)
    return output_tensors, weight_dict, faulty_indices, float32_bit_position, golden_bit_value, faulty_bit_value, is_signed

def execute_node(node, 
                 main_graph, 
                 final_output_node, 
                 weight_dict, 
                 inject_fault, 
                 inject_type, 
                 faulty_tensor_name, 
                 faulty_quantizer_name, 
                 faulty_bit_position, 
                 dequantized_operation_input_name, 
                 weight_tensor_name,
                 bias_output_name,
                 faulty_indices,
                 faulty_operation_name,
                 float32_bit_position,
		 golden_bit_value,
		 faulty_bit_value,
		 is_signed):
    node_inputs = []
    node_outputs = []

    added_quant_inputs, added_quant_outputs = expand_node_inputs_outputs(main_graph, node)
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

    if (inject_fault) and ("RANDOM" not in inject_type) and (node.op_type == "DequantizeLinear") and (faulty_quantizer_name in node.name):
        print(node.name)
        weight_dict, dequantized_operation_input_name, faulty_indices, golden_bit_value, faulty_bit_value, is_signed = perturb_quantizer(model, input_dict, weight_dict, faulty_tensor_name, faulty_bit_position)

    if "RANDOM" in inject_type:
        if (inject_fault) and (faulty_operation_name in node.name):
            faulty_value = None
            target_indices = [np.random.randint(0, dim) for dim in weight_dict[faulty_tensor_name].shape]
            golden_value = weight_dict[faulty_tensor_name][tuple(target_indices)]
            if "BITFLIP" in inject_type:
                faulty_value, float32_bit_position = float32_bit_flip(weight_dict[faulty_tensor_name], target_indices)
            else:
                faulty_value = delta_init(True)
            weight_dict[faulty_tensor_name][tuple(target_indices)] = faulty_value
            faulty_indices = target_indices

    if "INPUT" in inject_type or "WEIGHT" in inject_type:
        if (inject_fault) and (node.op_type == "Conv") and (node.name == faulty_operation_name):
            if not (dequantized_operation_input_name):
                sys.exit(0)
            delta_perturb = perturb_conv(model, input_dict, weight_dict, dequantized_operation_input_name, bias_output_name)

            if "16" in inject_type:
                delta_16 = np.zeros(delta_perturb.shape, dtype=delta_perturb.dtype)
                dim_batch, dim_channel, dim_height, dim_width = delta_perturb.shape
                positions_16 = []
                if "INPUT" in inject_type:
                    pad_y, pad_x = None, None
                    stride_y, stride_x = None, None
                    _, _, filter_dim_y, filter_dim_x = tuple(weight_dict[weight_tensor_name].shape)
                    for attribute in (node.attribute):
                        if (attribute.name) == "pads":
                            pad_x, pad_y, _, _ = (tuple((attribute).ints))
                        if (attribute.name) == "strides":
                            stride_x, stride_y = (tuple((attribute).ints))
                    if (pad_x is None) or (pad_y is None) or (stride_x is None) or (stride_y is None):
                        print("ERROR1")
                        sys.exit(0)
                    start_y, start_x = faulty_indices[2], faulty_indices[3]
                    start_y = np.random.randint(max(0, ((start_y + pad_y)//stride_y - filter_dim_y + 1)), min(((start_y + pad_y)//stride_y + 1), dim_height))
                    start_x = np.random.randint(max(0, ((start_x + pad_x)//stride_x - filter_dim_x + 1)), min(((start_x + pad_x)//stride_x + 1), dim_width))
                    start_channel = np.random.randint(dim_channel // 16)
                    for channel in range(16):
                        delta_16[0][start_channel + channel][start_y][start_x] = delta_perturb[0][start_channel + channel][start_y][start_x]
                        positions_16.append((0, 16*start_channel+channel, start_y, start_x))
                    delta_perturb = delta_16
                if "WEIGHT" in inject_type:
                    start_position = np.random.randint((dim_height * dim_width)//16)
                    start_channel = faulty_indices[0]
                    for inject_index in range(16):
                        if start_position * 16 + inject_index >= dim_height * dim_width:
                            break
                        inject_height = (start_position*16 + inject_index) // dim_width
                        inject_width = (start_position*16 + inject_index) % dim_width
                        delta_16[0][start_channel][inject_height][inject_width] = delta_perturb[0][start_channel][inject_height][inject_width]
                        positions_16.append((0, start_channel, inject_height, inject_width))
                    delta_perturb = delta_16

            perturb_result = np.add(original_tensor_output,delta_perturb)
            output_tensors[tensor_output_name] = perturb_result
            weight_dict[tensor_output_name] = perturb_result

    return output_tensors, weight_dict, dequantized_operation_input_name, faulty_indices, float32_bit_position, golden_bit_value, faulty_bit_value, is_signed

def fault_injection(original_weight_dict, main_graph, x_test, y_test, inject_type, faulty_tensor_name, faulty_quantizer_name, faulty_bit_position, weight_tensor_name, bias_output_name, faulty_operation_name, directory_name, model_specific_tensor_name, temp):
    faulty_indices = None
    inject_fault = False

    image_id = np.random.randint(0,10000)
    if "inception" in directory_name:
        input_data = preprocess_cifar10_inception(x_test[image_id])
    else:
        input_data = preprocess_cifar10(x_test[image_id])
    input_data = (np.expand_dims(input_data.numpy(), axis=0))
    original_weight_dict["global_in"] = input_data

    weight_dict = original_weight_dict.copy()
    output_tensors, weight_dict, faulty_indices, float32_bit_position, golden_bit_value, faulty_bit_value, is_signed = inference(main_graph, weight_dict, inject_fault, inject_type, faulty_tensor_name, faulty_quantizer_name, faulty_bit_position, weight_tensor_name, bias_output_name, faulty_indices, faulty_operation_name)
    golden_outputs = list((output_tensors[list(output_tensors.keys())[0]])[0])
    golden_output = str(golden_outputs.index(max(golden_outputs)))
    #specific_tensor_golden = copy.deepcopy(weight_dict[model_specific_tensor_name])

    inject_fault = True

    output_tensors = None
    weight_dict = original_weight_dict.copy()
    output_tensors, weight_dict, faulty_indices, float32_bit_position, golden_bit_value, faulty_bit_value, is_signed = inference(main_graph, weight_dict, inject_fault, inject_type, faulty_tensor_name, faulty_quantizer_name, faulty_bit_position, weight_tensor_name, bias_output_name, faulty_indices, faulty_operation_name)
    faulty_outputs = list((output_tensors[list(output_tensors.keys())[0]])[0])
    softmax = torch.nn.Softmax(dim=0)
    softmax_list_faulty = [tensor.item() for tensor in (softmax(torch.FloatTensor(faulty_outputs)))]
    softmax_list_golden = [tensor.item() for tensor in (softmax(torch.FloatTensor(golden_outputs)))]
    faulty_output = str(faulty_outputs.index(max(faulty_outputs)))
    #specific_tensor_faulty = copy.deepcopy(weight_dict[model_specific_tensor_name])
    weight_dict = original_weight_dict.copy()

    #total_different_bits, total_different_indices, total_output_indices = total_bits_diff(specific_tensor_golden, specific_tensor_faulty)

    validation_output = str(y_test[image_id][0])

    is_invalid = validation_output != golden_output
    is_mismatch = faulty_output != golden_output
    layer_name = str(faulty_operation_name)
    inject_type = str(inject_type)
    bit_position = str(faulty_bit_position)

    if float32_bit_position:
        bit_position = str(float32_bit_position)
    faulty_indices_string = "" 
    for index in faulty_indices:
        faulty_indices_string += str(index) + ";"
    faulty_indices_string = faulty_indices_string[:-1]

    output_string = str(is_invalid) + "," + str(is_mismatch) + "," + layer_name + "," + inject_type + "," + bit_position + "," + faulty_indices_string + "," + golden_output + "," + faulty_output + "," + str('{0:.3f}'.format(softmax_list_golden[int(golden_output)])) + "," + str('{0:.3f}'.format(softmax_list_faulty[int(golden_output)])) + "," + str(golden_bit_value) + "," + str(faulty_bit_value) + "," + str(is_signed) + "\n" #+ str(total_different_bits) + "," + str(total_different_indices) + "," + str(total_output_indices) + "\n"

    print("DATA:")
    print(validation_output)
    print(golden_output)
    print(faulty_output)
    print(output_string)

    with open("./results/models/" + directory_name + ".csv", "a") as file:
        file.write(output_string)

if __name__ == "__main__":
    _, (x_test, y_test) = cifar10.load_data()

    directory = "data/input/"
    directory_name = str(args.directory_name)
    directory = directory + directory_name
    bit_width = int(args.bit_width)
    dir_list = os.listdir(directory)

    for layer in dir_list:
        #for fault_model in ["INPUT", "WEIGHT", "INPUT16", "WEIGHT16", "RANDOM", "RANDOM_BITFLIP"]:
        for fault_model in ["WEIGHT", "WEIGHT16"]:
            image_id = np.random.randint(0, 10000)
            input_json_data = json.load(open(directory + "/" + layer))
            model_name = str(input_json_data["model_name"])
            input_tensor_name = str(input_json_data["input_tensor"])
            weight_tensor_name = str(input_json_data["weight_tensor"])
            bias_tensor_name = str(input_json_data["bias_tensor"])
            output_tensor_name = str(input_json_data["output_tensor"])
            #model_specific_tensor_name = str(input_json_data["target_tensor"])
            model_specific_tensor_name = None

            if "resnet" in model_name:
                model_name = "./assets/resnet50_w8a8.onnx"
                number_of_experiments = 60 
            else:
                model_name = "./assets/models/inceptionv1_w8a8.onnx"
                number_of_experiments = 15 

            faulty_operation_name = str(input_json_data["target_layer"])

            iterations = bit_width
            if "RANDOM" in fault_model:
                iterations = 2
            for bit_position in range(7,iterations):
                inject_type = str(fault_model)
                faulty_bit_position = None
                if "RANDOM" not in inject_type:
                    faulty_bit_position = int(bit_position)
                bias_output_name = bias_tensor_name

                if ("RANDOM" not in inject_type and faulty_bit_position > bit_width) or image_id >= 10000:
                    print("ERROR2")
                    sys.exit(0)

                model = ModelWrapper(model_name)
                main_graph = model.graph
                weights = model.graph.initializer
                weight_dict = {}
                for weight in weights:
                    weight_dict[weight.name] = numpy_helper.to_array(weight)
                
                if "inception" in directory_name:
                    input_data = preprocess_cifar10_inception(x_test[image_id])
                else:
                    input_data = preprocess_cifar10(x_test[image_id])
                input_data = (np.expand_dims(input_data.numpy(), axis=0))
                weight_dict["global_in"] = input_data

                (input_quantizer_name, int_input_tensor_name), (weight_quantizer_name, int_weight_tensor_name), (bias_quantizer_name, int_bias_tensor_name) = get_target_inputs(main_graph, faulty_operation_name, input_tensor_name, weight_tensor_name, bias_tensor_name, output_tensor_name)

                original_weight_tensor_name = weight_tensor_name
                weight_tensor_name = int_weight_tensor_name 
                if "INPUT" in inject_type:
                    faulty_quantizer_name = input_quantizer_name 
                    faulty_tensor_name = int_input_tensor_name 
                elif "WEIGHT" in inject_type:
                    faulty_quantizer_name = weight_quantizer_name 
                    faulty_tensor_name = int_weight_tensor_name
                elif "RANDOM" in inject_type:
                    faulty_quantizer_name = None
                    faulty_tensor_name = output_tensor_name 

                original_weight_dict = weight_dict.copy()

                with Pool() as pool:
                    start_time = time.time()
                    results = pool.map(partial(fault_injection, original_weight_dict, main_graph, x_test, y_test, inject_type, faulty_tensor_name, faulty_quantizer_name, faulty_bit_position, weight_tensor_name, bias_output_name, faulty_operation_name, directory_name, model_specific_tensor_name), [0 for n in range(number_of_experiments)])
                    print("TIME:")
                    print(time.time() - start_time)
                weight_tensor_name = original_weight_tensor_name
