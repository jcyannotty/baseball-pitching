import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Coupling layer - RealNVP (https://arxiv.org/pdf/1605.08803)
class CouplingLayer(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.scale_net = nn.Sequential(
            nn.Linear(dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim // 2),
            nn.Tanh()
        )
        self.translate_net = nn.Sequential(
            nn.Linear(dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim // 2)
        )

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        s = self.scale_net(x1) # scale
        t = self.translate_net(x1) # translation
        
        # Split the data
        y1 = x1
        y2 = x2 * torch.exp(s) + t
        
        log_det = s.sum(dim=1) # diagonal jacobian
        return torch.cat([y1, y2], dim=1), log_det

    def inverse(self, y):
        y1, y2 = y.chunk(2, dim=1)
        s = self.scale_net(y1)
        t = self.translate_net(y1)
        x1 = y1
        x2 = (y2 - t) * torch.exp(-s)
        log_det = -s.sum(dim=1)
        return torch.cat([x1, x2], dim=1), log_det


class NormalizingFlow(nn.Module):
    def __init__(self, dim, hidden_dim, n_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([CouplingLayer(dim, hidden_dim) for _ in range(n_layers)])
        self.base_dist = torch.distributions.MultivariateNormal(
            torch.zeros(dim), torch.eye(dim)
        )

    def forward(self, x):
        """Maps x → z and accumulates log-det"""
        log_det_sum = torch.zeros(x.size(0), device=x.device)
        z = x
        for layer in reversed(self.layers):
            z, log_det = layer.inverse(z)
            log_det_sum += log_det
        return z, log_det_sum

    def log_prob(self, x):
        z, log_det = self.forward(x)
        return self.base_dist.log_prob(z) + log_det

    def sample(self, n):
        z = self.base_dist.sample((n,))
        x = z
        for layer in self.layers: 
            x, _ = layer(x)
        return x

