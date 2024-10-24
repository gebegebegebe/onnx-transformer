from attention import MultiHeadedAttention
import torch
from model import make_model
from layer_norm import LayerNorm
import torch.nn as nn

@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5):
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
        print("FOO")

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
        linears_3 = layer_name + ".self_attn.linears.3"
        norm_1 = layer_name + ".sublayer.1.norm"
        target_keys = [norm_0, linears_0, linears_1, linears_2, linears_3, norm_1]
        return get_target_dict(target_keys)
            
    for name, module in model.named_modules():
        print(name)
        if isinstance(module, MultiHeadedAttention): 
            if ("decoder" not in name):
                target_ops = get_layer_ops_encoder(name[:-10])

                attn_ln = [target_ops[key] for key in list(target_ops.keys()) if ".sublayer.0.norm" in key][0]
                qkv = [ 
                    [target_ops[key] for key in list(target_ops.keys()) if ".linears.0" in key][0],
                    [target_ops[key] for key in list(target_ops.keys()) if ".linears.1" in key][0],
                    [target_ops[key] for key in list(target_ops.keys()) if ".linears.2" in key][0],
                ]
                name = "encoder.layers.0"
                qkv_input_scales = scales[name + ".self_attn.linears.0"]
                smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)

def main():
    vocab_src, vocab_tgt = load_vocab()
    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    act_scales = torch.load("scales/transformer_scales.pt")
    smooth_lm(model, act_scales)

if __name__ == "__main__":
    main()
