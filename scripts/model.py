import torch
import torch.nn as nn

class PollutionPINN(nn.Module):
    def __init__(self, hidden_dim=64):
        super(PollutionPINN, self).__init__()
        
        # Neural network layers: 3 inputs (x, y, t) -> predicted smoke concentration
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, y, t):
        # Stack coordinates into a single vector [x, y, t]
        inputs = torch.cat([x, y, t], dim=1)
        return self.net(inputs)