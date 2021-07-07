from numpy.core.numeric import Infinity
import torch as t
import torch.nn as nn


class MTCut(nn.Module):
    def __init__(self, d_model: int=128, n_head: int=8, num_layers: int=3, dropout: float=0.2):
        super(MTCut, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head)
        self.encoding_layer = nn.Sequential(
            nn.TransformerEncoder(encoder_layer, num_layers=num_layers),
            nn.Linear(in_features=d_model, out_features=64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.rerank = nn.Sequential(
            nn.Linear(in_features=64, out_features=32),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=32, out_features=1)
        )
        self.decison_layer = nn.Sequential(
            nn.Linear(in_features=64, out_features=1),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        x = self.encoding_layer(x)
        y1 = self.rerank(x)
        y2 = self.decison_layer(x)
        return y1, y2


if __name__ == '__main__':
    input = t.randn(5, 300, 128)
    model = MTCut()
    result = model(input)
    print(result[0].shape)  # (5, 300, 1)
    print(result[1][:2])