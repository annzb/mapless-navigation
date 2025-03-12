import torch
import torch.nn as nn


class TrainedDownsampling(nn.Module):
    def __init__(self, num_points, retain_fraction=0.5):
        """
        Args:
            retain_fraction (float): Fraction of points to retain during dropout (0 < retain_fraction <= 1).
        """
        super(TrainedDownsampling, self).__init__()
        self.retain_fraction = retain_fraction
        self.dropout_weights = nn.Parameter(torch.rand(num_points))

    def forward(self, points):
        """
        Forward pass to downsample points based on learned dropout probabilities.
        Args:
            points (torch.Tensor): Input points of shape [B, N, C].

        Returns:
            torch.Tensor: Downsampled points of shape [B, num_retain, C].
        """
        batch_size, num_points, num_features = points.shape
        retain_probabilities = torch.sigmoid(self.dropout_weights)  # [N]
        num_retain = max(1, int(num_points * self.retain_fraction))  # At least 1 point retained

        # Expand retain probabilities to match batch size
        batch_retain_probabilities = retain_probabilities.unsqueeze(0).expand(batch_size, -1)  # [B, N]

        # Perform top-k selection across the batch
        _, indices = torch.topk(batch_retain_probabilities, num_retain, dim=1, largest=True, sorted=False)  # [B, num_retain]

        # Gather points based on top-k indices
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, num_features)  # [B, num_retain, C]
        downsampled_points = torch.gather(points, 1, indices_expanded)  # [B, num_retain, C]

        return downsampled_points
