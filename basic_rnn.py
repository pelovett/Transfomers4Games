import torch
import torch.nn as nn
import torch.nn.functional as F

from math import floor

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.lin1 = nn.Linear(hidden_size, floor(hidden_size/2))
        self.lin2 = nn.Linear(floor(hidden_size/2), output_size)
        self.embed_size = floor(hidden_size/2)

    def get_embed_size(self):
        return self.embed_size

    
    def forward(self, input_vec, lengths):
        packed_input = nn.utils.rnn.pack_padded_sequence(input_vec, lengths, 
                            batch_first=True, enforce_sorted=False)
        packed_out, _ = self.rnn(packed_input)
        temp, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        temp = F.relu(self.lin1(temp))
        return self.lin2(temp)


    def initHidden(self, batch_size=1):
        return torch.zeros((batch_size, self.hidden_size), dtype=torch.double)

    
    def embed(self, input_vec):
        temp, _ = self.rnn(input_vec)
        return F.relu(self.lin1(temp))
