import torch.nn as nn

class TaskC(nn.Module):
    def __init__(self, d_model: int=128) -> None:
        super(TaskC, self).__init__()
        self.classification_layer = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        out = self.classification_layer(x)
        return out