import torch as t
import torch.nn as nn


class MtChoopy(nn.Module):
    def __init__(self, seq_len: int=300, d_model: int=128, n_head: int=8, num_layers: int=3, num_tasks: float=3, dropout: float=0.4):
        super(MtChoopy, self).__init__()
        self.seq_len = seq_len
        self.num_tasks = num_tasks
        self.position_encoding = nn.Parameter(t.randn(self.seq_len, 127), requires_grad=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dropout=dropout)
        self.encoding_layer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classi = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=1),
            nn.Sigmoid()
        )
        self.rerank = nn.Linear(in_features=d_model, out_features=1)
        self.decison_layer = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        pe = self.position_encoding.expand(x.shape[0], self.seq_len, 127)
        x = t.cat((x, pe), dim=2)
        x = self.encoding_layer(x)
        y0 = self.classi(x)
        y1 = self.rerank(x)
        y2 = self.decison_layer(x)
        if self.num_tasks == 3: return [y0, y1, y2]
        elif self.num_tasks == 2.1: return [y0, y2]
        else: return [y1, y2]


if __name__ == '__main__':
    input = t.randn(5, 40, 1)
    model = MtChoopy(seq_len=40)
    result = model(input)
    print(result[0].shape)  # (5, 40, 1)
    print(result[1][:2])