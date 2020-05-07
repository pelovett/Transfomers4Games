import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerModel(nn.Module):

    def __init__(self, output_size, input_size, nhead, hidden_size, nlayers, dropout=0.2):
        super(TransformerModel, self).__init__()
        self.src_mask = None
        self.embed_size = input_size
        self.lin1 = nn.Linear(input_size, hidden_size)
        encoder_layers = TransformerEncoderLayer(hidden_size, nhead, hidden_size, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.decoder = nn.Linear(hidden_size, output_size)


    def _generate_square_subsequent_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def forward(self, src):
        src = src.permute(1, 0, 2)
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(src.shape[0]).to(device)
            self.src_mask = mask
        src = src * math.sqrt(self.embed_size)
        output = self.lin1(src)
        output = self.transformer_encoder(output, self.src_mask)
        output = self.decoder(output)
        return output.permute(1, 0, 2)

    
    def embed(self, src):
        src = src.permute(1, 0, 2)
        #if self.src_mask is None or self.src_mask.size(0) != len(src):
        #    device = src.device
       #     mask = self._generate_square_subsequent_mask(src.shape[0]).to(device)
        #    self.src_mask = mask
        src = src * math.sqrt(self.embed_size)
        output = self.lin1(src)
        output = self.transformer_encoder(output, self.src_mask)
        return output


