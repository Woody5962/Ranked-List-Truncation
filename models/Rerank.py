import torch.nn as nn

class TaskR(nn.Module):
    def __init__(self, d_model: int=128) -> None:
        super(TaskR, self).__init__()
        self.rerank_layer = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        out = self.rerank_layer(x)
        return out