import torch as t
import torch.nn as nn


class AttnCut(nn.Module):
    def __init__(self, input_size: int=5, d_model: int=256, n_head: int=4, num_layers: int=1):
        super(AttnCut, self).__init__()
        self.encoding_layer = nn.LSTM(input_size=input_size, hidden_size=128, num_layers=2,
                              batch_first=True, bidirectional=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head)
        self.attention_layer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.rerank = nn.Linear(in_features=d_model, out_features=300)
        self.decison_layer = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=1),
            nn.Softmax(dim=1)
        )
        
    
    def forward(self, x):
        x = self.encoding_layer(x)[0]
        x = self.attention_layer(x)
        y1 = self.rerank(x)
        y2 = self.decison_layer(x)
        return y1, y2


if __name__ == '__main__':
    input = t.randn(5, 300, 5)
    model = AttnCut()
    result = model(input)
    print(result.size())  # (5, 300, 1)
    print(result[:2])