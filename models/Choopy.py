import torch as t
import torch.nn as nn
from collections import OrderedDict


class Choopy(nn.Module):
    def __init__(self, d_model: int=128, n_head: int=8, num_layers: int=3):
        super(Choopy, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head)
        self.model = nn.Sequential(OrderedDict([
            ('transformer', nn.TransformerEncoder(encoder_layer, num_layers=num_layers)),
            ('fc', nn.Linear(in_features=d_model, out_features=1)),
            ('softmax', nn.Softmax(dim=1))
        ]))
        
    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    input = t.randn(5, 300, 128)
    model = Choopy()
    result = model(input)
    print(result.size())  # (5, 300, 1)
    print(result[:2])

        