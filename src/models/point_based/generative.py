import torch
from models.base import RadarOccupancyModel
from models.point_based.decoders import MlpDecoder
from models.point_based.pointnet import SinglePointEncoder


class Baseline(RadarOccupancyModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'generative_baseline_v1.0'
        num_features = 128
        encoded_cloud_size = 1024
        final_cloud_size = 4096
        
        self.encoder = SinglePointEncoder(output_size=encoded_cloud_size, output_dim=num_features)
        self.decoder = MlpDecoder(latent_dim=num_features * 2, output_size=final_cloud_size, output_dim=4)

    def forward(self, X, debug=False, **kwargs):
        flat_pts, batch_idx = X

        encoded_clouds = []
        for sample_idx in range(self.batch_size):
            cloud = flat_pts[batch_idx == sample_idx]
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
            return pred_clouds, probs, predicted_flat_indices
        return probs, predicted_flat_indices
