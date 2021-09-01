import torch as t
import torch.nn as nn
from collections import OrderedDict


class Choopy(nn.Module):
    def __init__(self, seq_len: int=300, d_model: int=128, n_head: int=8, num_layers: int=3, dropout=0.2):
        super(Choopy, self).__init__()
        self.seq_len = seq_len
        self.position_encoding = nn.Parameter(t.randn(self.seq_len, 127), requires_grad=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dropout=dropout)
        self.attention_layer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decison_layer = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        pe = self.position_encoding.expand(x.shape[0], self.seq_len, 127)
        x = t.cat((x, pe), dim=2)
        x = self.attention_layer(x)
        x = self.decison_layer(x)
        return x


if __name__ == '__main__':
    input = t.randn(5, 40, 1)
    model = Choopy(seq_len=40)
    result = model(input)
    print(result.size())  # (5, 40, 1)
    print(result[:2])

        