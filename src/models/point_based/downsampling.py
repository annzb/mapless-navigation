import torch
import torch.nn as nn


class TrainedDownsampling(nn.Module):
    def __init__(self, num_points, retain_fraction=0.5):
        """
        Args:
            num_points (int): Total number of input points.
            retain_fraction (float): Fraction of points to retain (0 < retain_fraction <= 1).
        """
        super(TrainedDownsampling, self).__init__()
        self.retain_fraction = retain_fraction
        self.num_retain = int(num_points * retain_fraction)  # Fixed number of points
        self.dropout_weights = nn.Parameter(torch.randn(num_points))  # Learnable importance weights

    def forward(self, points):
        """
        Args:
            points (Tensor): Input tensor of shape [B, N, C], where
                B = batch size,
                N = number of input points,
                C = feature dimension (x, y, z, prob).

        Returns:
            Tensor: Downsampled points of shape [B, num_retain, C].
        """
        batch_size, num_points, num_features = points.shape

        # Compute soft selection probabilities
        sampling_probs = torch.softmax(self.dropout_weights, dim=0)  # [N]

        # Expand to match batch size
        sampling_probs = sampling_probs.unsqueeze(0).expand(batch_size, -1)  # [B, N]

        # Select points using weighted sum (soft selection)
        weighted_points = points * sampling_probs.unsqueeze(-1)  # [B, N, C]

        # Normalize by the sum of weights to maintain scale
        sum_weights = sampling_probs.sum(dim=1, keepdim=True)  # [B, 1]
        downsampled_points = weighted_points / (sum_weights.unsqueeze(-1) + 1e-6)  # [B, N, C]

        # Keep only the most important `num_retain` points (sorted by weight)
        _, indices = torch.topk(sampling_probs, self.num_retain, dim=1, largest=True, sorted=True)  # [B, num_retain]
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, num_features)  # [B, num_retain, C]

        # Extract the top-ranked weighted points
        downsampled_points = torch.gather(downsampled_points, 1, indices_expanded)  # [B, num_retain, C]

        return downsampled_points
