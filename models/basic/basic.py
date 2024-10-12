from torch import nn
import logging

logger = logging.getLogger(__name__)

class Basic(nn.Module):

    def __init__(self, input_dim=2, output_dim=3):
        super(Basic, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        layer_1 = nn.Linear(input_dim, 32)
        activation_1 = nn.ReLU()
        layer_2 = nn.Linear(32, 64)
        activation_2 = nn.ReLU()
        layer_3 = nn.Linear(64, output_dim)

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

        results = self.model(x)

        logger.debug(f"Output shape: {results.shape}")
        return results
