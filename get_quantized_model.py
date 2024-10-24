from attention import MultiHeadedAttention
from position_feed_forward import PositionwiseFeedForward
import torch
from model import make_model
from layer_norm import LayerNorm
import torch.nn as nn
from quant_linear import W8A8Linear

@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, name, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    assert isinstance(ln, LayerNorm)
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.a_2.numel() == fc.in_features == act_scales.numel()

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )

    ln.a_2.div_(scales)
    ln.b_2.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

def load_vocab():
    vocab_src, vocab_tgt = torch.load("vocab.pt")
    print("Finished.\nVocabulary sizes:")
    print(len(vocab_src))
    print(len(vocab_tgt))
    return vocab_src, vocab_tgt


def smooth_lm(model, scales, alpha=0.5):
    def get_target_dict(target_keys):
        target_dict = {}
        for name, module in model.named_modules():
            if name in target_keys:
                target_dict[name] = module
        return target_dict

    def get_layer_ops_encoder(layer_name):
        layer_number = layer_name[-1]
        norm_0 = layer_name + ".sublayer.0.norm"
        linears_0 = layer_name + ".self_attn.linears.0"
        linears_1 = layer_name + ".self_attn.linears.1"
        linears_2 = layer_name + ".self_attn.linears.2"
        #linears_3 = layer_name + ".self_attn.linears.3"
        norm_1 = layer_name + ".sublayer.1.norm"
        w_1 = layer_name + ".feed_forward.w_1"
        target_keys = [norm_0, linears_0, linears_1, linears_2, norm_1, w_1]
        return get_target_dict(target_keys)
            
    def get_layer_ops_decoder_self_attn(layer_name):
        layer_number = layer_name[-1]
        norm_0 = layer_name + ".sublayer.0.norm"
        linears_0 = layer_name + ".self_attn.linears.0"
        linears_1 = layer_name + ".self_attn.linears.1"
        linears_2 = layer_name + ".self_attn.linears.2"
        #linears_3 = layer_name + ".self_attn.linears.3"
        target_keys = [norm_0, linears_0, linears_1, linears_2]#, linears_3]
        return get_target_dict(target_keys)
            
    def get_layer_ops_decoder_src_attn(layer_name):
        layer_number = layer_name[-1]
        norm_0 = layer_name + ".sublayer.1.norm"
        linears_0 = layer_name + ".src_attn.linears.0"
        linears_1 = layer_name + ".src_attn.linears.1"
        linears_2 = layer_name + ".src_attn.linears.2"
        #linears_3 = layer_name + ".src_attn.linears.3"
        norm_1 = layer_name + ".sublayer.2.norm"
        w_1 = layer_name + ".feed_forward.w_1"
        target_keys = [norm_0, linears_0, linears_1, linears_2, norm_1, w_1]
        return get_target_dict(target_keys)
            
    for name, module in model.named_modules():
        if isinstance(module, MultiHeadedAttention): 
            if ("encoder" in name):
                name = name[:-10]
                target_ops = get_layer_ops_encoder(name)

                attn_ln = [target_ops[key] for key in list(target_ops.keys()) if ".sublayer.0.norm" in key][0]
                qkv = [ 
                    [target_ops[key] for key in list(target_ops.keys()) if ".linears.0" in key][0],
                    [target_ops[key] for key in list(target_ops.keys()) if ".linears.1" in key][0],
                    [target_ops[key] for key in list(target_ops.keys()) if ".linears.2" in key][0],
                ]
                qkv_input_scales = scales[name + ".self_attn.linears.0"]
                smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, name, alpha)

                ffn_ln = [target_ops[key] for key in list(target_ops.keys()) if ".sublayer.1.norm" in key][0],
                fc1 = [target_ops[key] for key in list(target_ops.keys()) if ".feed_forward.w_1" in key][0],
                fc1_input_scales = scales[name + ".feed_forward.w_1"]
                if isinstance(ffn_ln, tuple):
                    ffn_ln = ffn_ln[0]
                if isinstance(fc1, tuple):
                    fc1 = fc1[0]
                smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, name, alpha)
                continue

            elif ("decoder" in name) and (".self_attn" in name):
                name = name[:16]
                target_ops = get_layer_ops_decoder_self_attn(name)

                attn_ln = [target_ops[key] for key in list(target_ops.keys()) if ".sublayer.0.norm" in key][0]
                qkv = [ 
                    [target_ops[key] for key in list(target_ops.keys()) if ".linears.0" in key][0],
                    [target_ops[key] for key in list(target_ops.keys()) if ".linears.1" in key][0],
                    [target_ops[key] for key in list(target_ops.keys()) if ".linears.2" in key][0],
                ]
                qkv_input_scales = scales[name + ".self_attn.linears.0"]
                smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, name, alpha)
                continue

            elif ("decoder" in name) and ("src_attn" in name):
                name = name[:16]
                target_ops = get_layer_ops_decoder_src_attn(name)

                attn_ln = [target_ops[key] for key in list(target_ops.keys()) if ".sublayer.1.norm" in key][0]
                qkv = [ 
                    [target_ops[key] for key in list(target_ops.keys()) if ".linears.0" in key][0],
                    [target_ops[key] for key in list(target_ops.keys()) if ".linears.1" in key][0],
                    [target_ops[key] for key in list(target_ops.keys()) if ".linears.2" in key][0],
                ]
                qkv_input_scales = scales[name + ".src_attn.linears.0"]
                smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, name, alpha)

                ffn_ln = [target_ops[key] for key in list(target_ops.keys()) if ".sublayer.2.norm" in key][0],
                fc1 = [target_ops[key] for key in list(target_ops.keys()) if ".feed_forward.w_1" in key][0],
                fc1_input_scales = scales[name + ".feed_forward.w_1"]
                if isinstance(ffn_ln, tuple):
                    ffn_ln = ffn_ln[0]
                if isinstance(fc1, tuple):
                    fc1 = fc1[0]
                smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, name, alpha)
                continue

def quantize_transformer(model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=True):
    for name, module in model.named_modules():
        if isinstance(module, PositionwiseFeedForward):
            module.w_1 = W8A8Linear.from_float(
                module.w_1, weight_quant=weight_quant, act_quant=act_quant
            ) 
            module.w_2 = W8A8Linear.from_float(
                module.w_2, weight_quant=weight_quant, act_quant=act_quant
            ) 
        elif isinstance(module, MultiHeadedAttention):
            module.linears[0] = W8A8Linear.from_float(
                module.linears[0], weight_quant=weight_quant, act_quant=act_quant, quantize_output=True
            ) 
            module.linears[1] = W8A8Linear.from_float(
                module.linears[1], weight_quant=weight_quant, act_quant=act_quant, quantize_output=True
            ) 
            module.linears[2] = W8A8Linear.from_float(
                module.linears[2], weight_quant=weight_quant, act_quant=act_quant, quantize_output=True
            ) 
            module.linears[3] = W8A8Linear.from_float(
                module.linears[3], weight_quant=weight_quant, act_quant=act_quant
            ) 
    return model

def main():
    import copy
    vocab_src, vocab_tgt = load_vocab()
    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(torch.load("checkpoint/iwslt14_model_final.pt", map_location=torch.device("cpu")))

    original_parameters = {}
    for name, parameter in model.named_parameters():
        original_parameters[name] = copy.deepcopy(parameter)

    act_scales = torch.load("scales/transformer_scales.pt")
    smooth_lm(model, act_scales)

    ptq_parameters = {}
    for name, parameter in model.named_parameters():
        ptq_parameters[name] = parameter 

    for name, parameter in model.named_parameters():
        if (("encoder" in name) or ("decoder" in name)) and ("bias" not in name):
            if (torch.equal(original_parameters[name], ptq_parameters[name])):
                print("--")
                print(name)

    print("FOOBARBAZ")
    model = quantize_transformer(model)
    print(model)

if __name__ == "__main__":
    main()
