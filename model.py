import torch
import torch.nn as nn
import torch.nn.functional as F

class Radar_LSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.LSTM = nn.LSTM(input_size=3, hidden_size=15, batch_first=True)
        self.linear_layers_post = [nn.Linear(15, 64),
                                  nn.Linear(64, 64)]

        self.linear_layers_post = nn.ModuleList(self.linear_layers_post)

    def forward(self, x):
        x, _ = self.LSTM(x)
        x = x[:, -1, :]
        
        for i in range(len(self.linear_layers_post)-1):
            x = self.linear_layers_post[i](x)
            x = F.relu(x)
        
        x = self.linear_layers_post[-1](x)

        return x
