import torch
import numpy as np
from collections import OrderedDict
from torch import nn
import logging

logger = logging.getLogger(__name__)


class Sine(nn.Module):
    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)


class SIREN(nn.Module):
    '''
    A fully connected neural network that uses sine 
    '''

    def __init__(
        self,
        in_features=2,
        out_features=1,
        num_hidden_layers=3,
        hidden_features=256
    ):
        super().__init__()

        nonlinearity = Sine()
        self.weight_init = sine_initialization

        self.net = []
        # Initialize the first layer 
        self.net.append(
            nn.Sequential(
                nn.Linear(in_features, hidden_features), nonlinearity
            ).apply(first_layer_sine_init)
        )

        for i in range(num_hidden_layers):
            self.net.append(
                nn.Sequential(
                    nn.Linear(hidden_features, hidden_features), nonlinearity
                )
            )

        self.net.append(
            nn.Sequential(
                nn.Linear(hidden_features, out_features)
            )
        )

        self.net = nn.Sequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        logger.info("Basic model initialized")

    def forward(self, model_input):
        # Enables us to compute gradients w.r.t. coordinates
        coords = model_input['coords'].clone().detach().requires_grad_(True)

        logger.debug(f"Input shape: {coords.shape}")
        output = self.net(coords)
        logger.debug(f"Output shape: {output.shape}")

        return {'model_in': coords, 'model_out': output}

    def forward_with_activations(self, model_input):
        '''Returns not only model output, but also intermediate activations.'''
        coords = model_input['coords'].clone().detach().requires_grad_(True)

        activations = OrderedDict()
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            for j, sublayer in enumerate(layer):
                x = sublayer(x)
                activations['_'.join((str(sublayer.__class__), "%d" % i))] = x
                
        return {'model_in': coords, 'model_out': activations.popitem(), 'activations': activations}

def sine_initialization(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)