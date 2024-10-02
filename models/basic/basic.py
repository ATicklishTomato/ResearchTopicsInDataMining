import torch
from torch import nn
import logging

logger = logging.getLogger(__name__)

class Basic(nn.Module):

    def __init__(self, input_dim=2, output_dim=3, output_classes=256):
        super(Basic, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_classes = output_classes

        layer_1 = nn.Linear(input_dim, 64)
        activation_1 = nn.ReLU()
        layer_2 = nn.Linear(64, 128)
        activation_2 = nn.ReLU()
        # output is RGB values, so we need 3 output nodes within the range [0, 256]
        layer_3 = nn.Linear(128, output_dim * output_classes)
        self.activation_3 = nn.Softmax()

        self.model = nn.Sequential(
            layer_1,
            activation_1,
            layer_2,
            activation_2,
            layer_3
        )

        logger.info("Basic model initialized")


    def forward(self, x):
        logger.debug(f"Input shape: {x.shape}")

        results = self.model(x).view(-1, self.output_dim, self.output_classes)
        results = self.activation_3(results)

        results = torch.argmax(results, dim=1)

        logger.debug(f"Output shape: {results.shape}")
        return results
