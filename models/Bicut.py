import torch as t
import torch.nn as nn


class BiCut(nn.Module):
    def __init__(self, input_size=231451, lstm_hiden_size=128, lstm_layers=2, lstm_dropout=0.4, fc_dimensions=256):
        super(BiCut, self).__init__()
        self.bilstm = nn.LSTM(input_size=input_size, hidden_size=lstm_hiden_size, num_layers=lstm_layers,
                              batch_first=True, dropout=lstm_dropout, bidirectional=True)
        self.fc = nn.Linear(in_features=lstm_hiden_size * 2, out_features=fc_dimensions)
        self.softmax = nn.Sequential(
            nn.Linear(in_features=fc_dimensions, out_features=2),
            nn.Softmax(dim=2)
        )

    def forward(self, x):
        x = self.bilstm(x)[0]
        x = nn.functional.relu(self.fc(x))
        return self.softmax(x)


if __name__ == '__main__':
    input = t.randn(5, 300, 5)
    model = BiCut(input_size=5)
    result = model(input)
    print(result.size())  # (5, 300, 2)
    print(result[:5])
