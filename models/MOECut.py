# coding: UTF-8
import torch
import torch.nn as nn


class Expert(nn.Module):
    def __init__(self, d_model, n_head, num_layers, dropout: float=0.2):
        super(Expert, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dropout=dropout)
        self.attention_layer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        out = self.attention_layer(x)
        return out
    

class TowerCut(nn.Module):
    def __init__(self, d_model):
        super(TowerCut, self).__init__()
        self.cut_layer = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        out = self.cut_layer(x)
        return out


class TowerClass(nn.Module):
    def __init__(self, d_model):
        super(TowerClass, self).__init__()
        self.classification_layer = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        out = self.classification_layer(x)
        return out


class TowerRerank(nn.Module):
    def __init__(self, d_model):
        super(TowerRerank, self).__init__()
        self.rerank_layer = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        out = self.rerank_layer(x)
        return out


class MOECut(torch.nn.Module): 
    def __init__(self, seq_len: int=300, num_experts=3, num_tasks=3, input_size=3, encoding_size=128, d_model=256, n_head=4, num_layers=1, dropout=0.2):
        super(MOECut, self).__init__()
        # params
        self.seq_len = seq_len
        self.expert_hidden = d_model
        # the first embedding layer
        self.pre_encoding = nn.LSTM(input_size=input_size, hidden_size=encoding_size, num_layers=2, batch_first=True, bidirectional=True)
        # row by row
        self.softmax = nn.Softmax(dim=1)
        # model
        self.experts = nn.ModuleList([Expert(self.expert_hidden, n_head, num_layers, dropout) for _ in range(num_experts)])
        self.w_gates = nn.Parameter(torch.randn(encoding_size * self.seq_len * 2, num_experts), requires_grad=True)
        if num_tasks == 3:
            self.towers = nn.ModuleList([
                TowerClass(self.expert_hidden),
                TowerRerank(self.expert_hidden),
                TowerCut(self.expert_hidden)
            ])
        elif num_tasks == 2.1:
            self.towers = nn.ModuleList([
                TowerClass(self.expert_hidden),
                TowerCut(self.expert_hidden)
            ])
        elif num_tasks == 2.2:
            self.towers = nn.ModuleList([
                TowerRerank(self.expert_hidden),
                TowerCut(self.expert_hidden)
            ])

    def forward(self, x):
        # get the experts output
        experts_in = self.pre_encoding(x)[0]
        experts_o = [e(experts_in) for e in self.experts]
        experts_o_tensor = torch.stack(experts_o)

        # get the gates output
        batch_size = experts_in.shape[0]
        gates_o = self.softmax(experts_in.reshape(batch_size, -1) @ self.w_gates)

        # multiply the output of the experts with the corresponding gates output
        # res = gates_o[0].t().unsqueeze(2).expand(-1, -1, self.experts_out) * expers_o_tensor
        # https://discuss.pytorch.org/t/element-wise-multiplication-of-the-last-dimension/79534
        # 每条输入数据（self.seq_len * hidden）都对应一组权重
        towers_input = gates_o.t().unsqueeze(2).expand(-1, -1, self.seq_len).unsqueeze(3).expand(-1, -1, -1, self.expert_hidden) * experts_o_tensor
        towers_input = torch.sum(towers_input, dim=0)

        # get the final output from the towers
        final_output = [t(towers_input) for t in self.towers]

        # get the output of the towers, and stack them
        # final_output = torch.stack(final_output, dim=1)

        return final_output
    
if __name__ == '__main__':
    input = torch.randn(5, 40, 3)
    model = MOECut(seq_len=40)
    result = model(input)
    print(result[0].shape)  # (5, 300, 1)
    print(result[1][:2])