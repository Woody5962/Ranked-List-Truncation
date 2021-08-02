import torch as t
import torch.nn as nn

class MtAttnCut(nn.Module):
    def __init__(self, input_size: int=3, d_model: int=256, n_head: int=4, num_layers: int=1, num_tasks: float=3, dropout: float=0.4):
        super(MtAttnCut, self).__init__()
        self.num_tasks = num_tasks
        self.pre_encoding = nn.LSTM(input_size=input_size, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
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
        x = self.pre_encoding(x)[0]
        x = self.encoding_layer(x)
        y0 = self.classi(x)
        y1 = self.rerank(x)
        y2 = self.decison_layer(x)
        if self.num_tasks == 3: return [y0, y1, y2]
        elif self.num_tasks == 2.1: return [y0, y2]
        else: return [y1, y2]


if __name__ == '__main__':
    input = t.randn(5, 300, 3)
    model = MtAttnCut()
    result = model(input)
    print(result[0].shape)  # (5, 300, 1)
    print(result[1][:2])