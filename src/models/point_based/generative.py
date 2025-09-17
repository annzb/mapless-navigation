from typing import Optional

import torch
from models.base import RadarOccupancyModel
from models.point_based.decoders import MlpDecoder
from models.point_based.pointnet import SinglePointEncoder


class Baseline(RadarOccupancyModel):
    def __init__(
            self, 
            encoder_cloud_size: int,
            encoder_num_features: int, 
            encoder_batch_norm: bool, 
            encoder_dropout: Optional[float], 
            predicted_cloud_size: int,
            decoder_layer_norm: bool,
            decoder_dropout: Optional[float],
            **kwargs
        ):
        super().__init__(**kwargs)
        self.name = 'generative_baseline_v1.0'
        self.encoder = SinglePointEncoder(output_size=encoder_cloud_size, output_dim=encoder_num_features, batch_norm=encoder_batch_norm, dropout=encoder_dropout)
        self.decoder = MlpDecoder(latent_dim=encoder_num_features * 2, output_size=predicted_cloud_size, output_dim=4, layer_norm=decoder_layer_norm, dropout=decoder_dropout)
# X
# cloud, encoded, pooled_mean, pooled_max, latent, encoded_clouds
# pred_clouds, predicted_log_odds_flat, probs, predicted_flat_indices
    def forward(self, X, debug=False, **kwargs):
        flat_pts, batch_idx = X

        encoded_clouds = []
        for sample_idx in range(self.batch_size):
            cloud = flat_pts[batch_idx == sample_idx]
            if cloud.shape[0] == 0: break  # last batch may not be full

            encoded = self.encoder(cloud)
            pooled_mean = encoded.mean(dim=0, keepdim=True)
            pooled_max = encoded.max(dim=0, keepdim=True).values
            latent = torch.cat([pooled_mean, pooled_max], dim=-1)
            encoded_clouds.append(latent)
        encoded_clouds = torch.stack(encoded_clouds, dim=0)

        pred_clouds = self.decoder(encoded_clouds)

        predicted_log_odds_flat, predicted_flat_indices = self.merge_batches(pred_clouds)
        self.check_gradient(predicted_log_odds_flat, "Merged predictions")

        probs = self.apply_sigmoid(predicted_log_odds_flat)
        self.check_gradient(probs, "Final probabilities")

        if debug:
            return pred_clouds, predicted_log_odds_flat, probs, predicted_flat_indices
        return probs, predicted_flat_indices
