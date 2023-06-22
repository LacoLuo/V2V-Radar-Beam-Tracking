import torch
import torch.nn as nn
import torch.nn.functional as F

class RD_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d((2, 2))
        self.conv_layers = [nn.Conv2d(1, 8, 3, padding='same'),
                            nn.Conv2d(8, 16, 3, padding='same'),
                            nn.Conv2d(16, 8, 3, padding='same'),
                            nn.Conv2d(8, 4, 3, padding='same'),
                            nn.Conv2d(4, 2, 3, padding='same')]

        self.conv_pools = torch.zeros(len(self.conv_layers), dtype=bool)
        self.conv_pools[-4:] = True

        self.linear_layers_pre = [nn.Linear(256, 4*64),
                                 nn.Linear(4*64, 2*64)]
        

        self.LSTM = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)

        self.linear_layers_post = [nn.Linear(128+64, 128),
                                   nn.Linear(128, 128),
                                   nn.Linear(128, 64)]

        self.conv_layers = nn.ModuleList(self.conv_layers)
        self.linear_layers_pre = nn.ModuleList(self.linear_layers_pre)
        self.linear_layers_post = nn.ModuleList(self.linear_layers_post)

        self.embed = nn.Embedding(64, 64)

    def forward(self, x, beam):
        batch_size = x.shape[0]
        time_steps = x.shape[1]
        input_size = x.shape[2:]
        
        td_concat_size = (batch_size*time_steps,) + input_size
        
        beam = self.embed(beam).squeeze(1)

        x = x.view(td_concat_size)

        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x)
            if self.conv_pools[i]:
                x = self.pool(x)
            x = F.relu(x)
        
        x = torch.flatten(x, start_dim=1) ## [batch_size*time_steps, 256]

        for i in range(len(self.linear_layers_pre)):
            x = self.linear_layers_pre[i](x)
            x = F.relu(x)

        x = x.view(batch_size, time_steps, -1)
        x, _ = self.LSTM(x)
        x = x[:, -1, :]
        
        x = torch.cat((x, beam), dim=1)

        for i in range(len(self.linear_layers_post)-1):
            x = self.linear_layers_post[i](x)
            x = F.relu(x)
        x = self.linear_layers_post[-1](x)
        return x

class RA_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d((2, 2))

        self.conv_layers = [nn.Conv2d(1, 8, 3, padding='same'),
                            nn.Conv2d(8, 16, 3, padding='same'),
                            nn.Conv2d(16, 8, 3, padding='same'),
                            nn.Conv2d(8, 4, 3, padding='same'),
                            nn.Conv2d(4, 2, 3, padding='same')]

        self.conv_pools = torch.zeros(len(self.conv_layers), dtype=bool)
        self.conv_pools[-3:] = True

        self.linear_layers_pre = [nn.Linear(512, 4*64),
                                 nn.Linear(4*64, 2*64)]

        self.LSTM = nn.LSTM(input_size=128, hidden_size=128, batch_first=True, dropout=0.5)

        self.linear_layers_post = [nn.Linear(128+64, 128),
                                   nn.Linear(128, 128),
                                   nn.Linear(128, 64)]

        self.conv_layers = nn.ModuleList(self.conv_layers)
        self.linear_layers_pre = nn.ModuleList(self.linear_layers_pre)
        self.linear_layers_post = nn.ModuleList(self.linear_layers_post)

        self.embed = nn.Embedding(64, 64)

    def forward(self, x, beam):
        batch_size = x.shape[0]
        time_steps = x.shape[1]
        input_size = x.shape[2:]
         
        td_concat_size = (batch_size*time_steps,) + input_size
        
        beam = self.embed(beam).squeeze(1)

        x = x.view(td_concat_size)

        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x)
            if self.conv_pools[i]:
                x = self.pool(x)
            x = F.relu(x)
        
        x = torch.flatten(x, start_dim=1) ## [batch_size*time_steps, 256]

        for i in range(len(self.linear_layers_pre)):
            x = self.linear_layers_pre[i](x)
            x = F.relu(x)

        x = x.view(batch_size, time_steps, -1)
        x, _ = self.LSTM(x)
        x = x[:, -1, :]
        
        x = torch.cat((x, beam), dim=1)

        for i in range(len(self.linear_layers_post)-1):
            x = self.linear_layers_post[i](x)
            x = F.relu(x)
        x = self.linear_layers_post[-1](x)
        return x

class DA_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d((2, 2))

        self.conv_layers = [nn.Conv2d(1, 8, 3, padding='same'),
                            nn.Conv2d(8, 16, 3, padding='same'),
                            nn.Conv2d(16, 8, 3, padding='same'),
                            nn.Conv2d(8, 4, 3, padding='same'),
                            nn.Conv2d(4, 2, 3, padding='same')]

        self.conv_pools = torch.zeros(len(self.conv_layers), dtype=bool)
        self.conv_pools[-3:] = True

        self.linear_layers_pre = [nn.Linear(256, 4*64),
                                 nn.Linear(4*64, 2*64)]

        self.LSTM = nn.LSTM(input_size=128, hidden_size=128, batch_first=True, dropout=0.5)

        self.linear_layers_post = [nn.Linear(128+64, 128),
                                   nn.Linear(128, 128),
                                   nn.Linear(128, 64)]

        self.conv_layers = nn.ModuleList(self.conv_layers)
        self.linear_layers_pre = nn.ModuleList(self.linear_layers_pre)
        self.linear_layers_post = nn.ModuleList(self.linear_layers_post)

        self.embed = nn.Embedding(64, 64)

    def forward(self, x, beam):
        batch_size = x.shape[0]
        time_steps = x.shape[1]
        input_size = x.shape[2:]
         
        td_concat_size = (batch_size*time_steps,) + input_size
        
        beam = self.embed(beam).squeeze(1)

        x = x.view(td_concat_size)

        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x)
            if self.conv_pools[i]:
                x = self.pool(x)
            x = F.relu(x)
        
        x = torch.flatten(x, start_dim=1) ## [batch_size*time_steps, 256]

        for i in range(len(self.linear_layers_pre)):
            x = self.linear_layers_pre[i](x)
            x = F.relu(x)

        x = x.view(batch_size, time_steps, -1)
        x, _ = self.LSTM(x)
        x = x[:, -1, :]
        
        x = torch.cat((x, beam), dim=1)

        for i in range(len(self.linear_layers_post)-1):
            x = self.linear_layers_post[i](x)
            x = F.relu(x)
        x = self.linear_layers_post[-1](x)
        return x
