import torch
from torch import nn

from models.NFFB.FFB_encoder import FFB_encoder
import logging

logger = logging.getLogger(__name__)


class NFFB(nn.Module):
    def __init__(self, input_dims=2, out_dims=3):
        super().__init__()

        encoding_config = {
            "feat_dim": input_dims,
            "base_resolution": 64,  # FC layers
            "per_level_scale": 2,  # cg
            "base_sigma": 5.0,  # sigma min
            "exp_sigma": 2.0,  # cf
            "grid_embedding_std": 0.01,
        }

        network_config = {
            "dims": [64, 64, 64, 64],
            "w0": 100.0,
            "w1": 100.0,
            "size_factor": 1,
        }

        backbone = {"dims": [64, 64]}

        self.xyz_encoder = FFB_encoder(
            n_input_dims=input_dims,
            encoding_config=encoding_config,
            network_config=network_config,
            has_out=True,
        )

        ### Initializing backbone part, to merge multi-scale grid features
        backbone_dims = backbone["dims"]
        grid_feat_len = self.xyz_encoder.out_dim
        backbone_dims = [grid_feat_len] + backbone_dims + [out_dims]
        self.num_backbone_layers = len(backbone_dims)

        for layer in range(0, self.num_backbone_layers - 1):
            out_dim = backbone_dims[layer + 1]
            setattr(
                self,
                "backbone_lin" + str(layer),
                nn.Linear(backbone_dims[layer], out_dim),
            )

        self.relu_activation = nn.ReLU(inplace=True)
        
        logger.info("NFFB model initialized")

    @torch.no_grad()
    # optimizer utils
    def get_params(self, LR_schedulers):
        params = [{"params": self.parameters(), "lr": LR_schedulers[0]["initial"]}]

        return params

    def forward(self, x):
        """
        Inputs:
            x: (N, 2) xy in [-scale, scale]
        Outputs:
            out: (N, 1 or 3), the RGB values
        """
        logger.debug(f"Input shape: {x.shape}")
        
        if isinstance(x, dict):
            coords = x["coords"].clone().detach().requires_grad_(True)

        x = (coords - 0.5) * 2.0
        out_feat = self.xyz_encoder(x)

        ### Backbone transformation
        for layer in range(0, self.num_backbone_layers - 1):
            backbone_lin = getattr(self, "backbone_lin" + str(layer))
            out_feat = backbone_lin(out_feat)

            if layer < self.num_backbone_layers - 2:
                out_feat = self.relu_activation(out_feat)

        out_feat = out_feat.clamp(-1.0, 1.0)
        
        logger.debug(f"Output shape: {out_feat.shape}")

        return {"model_in": coords, "model_out": out_feat}
