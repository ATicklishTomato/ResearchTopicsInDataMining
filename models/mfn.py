import torch
from torch import nn
import numpy as np
import logging

logger = logging.getLogger(__name__)

class MFNBase(nn.Module):
    def __init__(self, hidden_size, out_size, n_layers, weight_scale, bias=True, output_act=False):
        super().__init__()
        self.linear = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias) for _ in range(n_layers)])
        logger.debug(f"Creating MFN with {n_layers} layers")
        self.output_linear = nn.Linear(hidden_size, out_size)
        self.output_act = output_act
        for lin in self.linear:
            lin.weight.data.uniform_(-np.sqrt(weight_scale / hidden_size), np.sqrt(weight_scale / hidden_size))
        logger.info(f"MFN model initialized")
    def forward(self, x):
        x = x["coords"].detach().clone().requires_grad_(True)
        logger.debug(f"MFN input shape: {x.shape}")
        out = self.filters[0](x)
        for i in range(1, len(self.filters)):
            out = self.filters[i](x) * self.linear[i - 1](out)
        out = self.output_linear(out)
        if self.output_act:
            out = torch.sin(out)
        logger.debug(f"MFN output shape: {out.shape}")
        output = {"model_in": x, 'model_out': out}
        return output
        
"""
class FourierLayer(nn.Module):
    def __init__(self, in_features, out_features, weight_scale):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.linear.weight.data *= weight_scale
        self.linear.bias.data.uniform_(-np.pi, np.pi)

    def forward(self, x):
        return torch.sin(self.linear(x))

class FourierNet(MFNBase):
    def __init__(self, in_size, hidden_size, out_size, n_layers=3, input_scale=256.0, weight_scale=1.0, bias=True, output_act=False):
        super().__init__(hidden_size, out_size, n_layers, weight_scale, bias, output_act)
        self.filters = nn.ModuleList([FourierLayer(in_size, hidden_size, input_scale / np.sqrt(n_layers + 1)) for _ in range(n_layers + 1)])
"""

class GaborLayer(nn.Module):
    def __init__(self, in_features, out_features, weight_scale, alpha=1.0, beta=1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.mu = nn.Parameter(2 * torch.rand(out_features, in_features) - 1)
        self.gamma = nn.Parameter(torch.distributions.gamma.Gamma(alpha, beta).sample((out_features,)))
        self.linear.weight.data *= weight_scale * torch.sqrt(self.gamma[:, None])
        self.linear.bias.data.uniform_(-np.pi, np.pi)
        logger.info(f"Gabor layer initialized with {in_features} input features and {out_features} output features")

    def forward(self, x):
        D = (x ** 2).sum(-1)[..., None] + (self.mu ** 2).sum(-1)[None, :] - 2 * x @ self.mu.T
        y = torch.sin(self.linear(x)) * torch.exp(-0.5 * D * self.gamma[None, :])
        output = {"model_in": x, 'model_out': y}
        logger.debug(f"Gabor layer output shape: {y.shape}")
        return y

class GaborNet(MFNBase):
    def __init__(self, in_size, hidden_size, out_size, n_layers=3, input_scale=256.0, weight_scale=1.0, alpha=6.0, beta=1.0, bias=True, output_act=False):
        super().__init__(hidden_size, out_size, n_layers, weight_scale, bias, output_act)
        self.filters = nn.ModuleList([GaborLayer(in_size, hidden_size, input_scale / np.sqrt(n_layers + 1), alpha / (n_layers + 1), beta) for _ in range(n_layers + 1)])
        logger.info(f"GaborNet model initialized")
