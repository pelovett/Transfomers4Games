import torch
from transformer_model import *

model = TransformerModel(4, 4, 5, 50, 4)
x = torch.rand((1,200,4))

model(x)
