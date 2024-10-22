import torch.nn as nn
from brevitas.export import export_onnx_qcdq

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        temp = self.encode(src, src_mask)
        print("FOCUS HERE")
        print(temp.shape)
        return self.decode(temp, src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def export_encoder(self, src, src_mask, export_path):
        encoder_input = (self.src_embed(src), src_mask)
        export_onnx_qcdq(self.encoder, encoder_input, export_path, opset_version=13)

    def export_decoder(self, memory, src_mask, tgt, tgt_mask, export_path):
        decoder_input = (self.tgt_embed(tgt), memory, src_mask, tgt_mask)
        export_onnx_qcdq(self.decoder, decoder_input, export_path, opset_version=13, dynamic_axes={
            "onnx::ReduceMean_0": [1],
            "onnx::Unsqueeze_3": [1,2],
        })

    def get_src_embed(self, src):
        return self.src_embed(src)

    def get_tgt_embed(self, tgt):
        return self.tgt_embed(tgt)
