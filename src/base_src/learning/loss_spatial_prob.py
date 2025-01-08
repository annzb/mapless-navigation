import torch
import torch.nn as nn
import torch.nn.functional as F


def match_pointclouds(true_xyz, pred_xyz, max_distance=float('inf')):
    """
    Matches true points to predicted points within a maximum distance.

    Args:
        true_xyz (torch.Tensor): Ground truth points of shape [N_true, 3].
        pred_xyz (torch.Tensor): Predicted points of shape [N_pred, 3].
        max_distance (float): Maximum allowable distance for matching.

    Returns:
        matched_true_xyz (torch.Tensor): Matched true points, or empty tensor if no matches.
        matched_pred_xyz (torch.Tensor): Matched predicted points, or empty tensor if no matches.
        matched_true_idx (torch.Tensor): Indices of matched true points, or empty tensor if no matches.
        matched_pred_idx (torch.Tensor): Indices of matched predicted points, or empty tensor if no matches.
    """
    if true_xyz.size(0) == 0 or pred_xyz.size(0) == 0:
        return (
            torch.empty((0, 3), device=true_xyz.device),
            torch.empty((0, 3), device=pred_xyz.device),
            torch.empty((0,), dtype=torch.long, device=true_xyz.device),
            torch.empty((0,), dtype=torch.long, device=pred_xyz.device),
        )

    dists = torch.cdist(true_xyz, pred_xyz)  # [N_true, N_pred]
    valid_mask = dists <= max_distance
    dists[~valid_mask] = float('inf')

    matched_true_idx = []
    matched_pred_idx = []
    for i in range(dists.size(0)):
        if valid_mask[i].any():  # Check if there are any valid matches for this true point
            min_dist, min_idx = dists[i].min(dim=0)
            if min_dist != float('inf'):  # Valid match found
                matched_true_idx.append(i)
                matched_pred_idx.append(min_idx.item())
                dists[:, min_idx] = float('inf')  # Invalidate the matched predicted point

    if not matched_true_idx:
        return (
            torch.empty((0, 3), device=true_xyz.device),
            torch.empty((0, 3), device=pred_xyz.device),
            torch.empty((0,), dtype=torch.long, device=true_xyz.device),
            torch.empty((0,), dtype=torch.long, device=pred_xyz.device),
        )
    matched_true_idx = torch.tensor(matched_true_idx, dtype=torch.long, device=true_xyz.device)
    matched_pred_idx = torch.tensor(matched_pred_idx, dtype=torch.long, device=pred_xyz.device)
    matched_true_xyz = true_xyz[matched_true_idx]
    matched_pred_xyz = pred_xyz[matched_pred_idx]
    return matched_true_xyz, matched_pred_xyz, matched_true_idx, matched_pred_idx


class SpatialProbLoss(nn.Module):
    def __init__(self, occupancy_threshold=0.5, point_match_radius=1.0):
        """
        Initializes the loss class with threshold and radius parameters.

        Args:
            occupancy_threshold (float): Threshold to filter occupied points based on probabilities.
            point_match_radius (float): Maximum distance for matching points.
        """
        super(SpatialProbLoss, self).__init__()
        self.occupancy_threshold = occupancy_threshold
        self.point_match_radius = point_match_radius

    def forward(self, pred_cloud, true_cloud):
        """
        Computes the spatial-probability loss.

        Args:
            pred_cloud (torch.Tensor): Predicted point cloud of shape [N_pred, 4] (XYZP).
            true_cloud (torch.Tensor): Ground truth point cloud of shape [N_true, 4] (XYZP).

        Returns:
            torch.Tensor: Combined spatial and probability loss.
        """
        pred_occupied = pred_cloud[pred_cloud[:, -1] >= self.occupancy_threshold]
        true_occupied = true_cloud[true_cloud[:, -1] >= self.occupancy_threshold]
        pred_xyz, true_xyz = pred_occupied[:, :3], true_occupied[:, :3]
        pred_probs, true_probs = pred_occupied[:, 3], true_occupied[:, 3]

        matched_true_xyz, matched_pred_xyz, matched_true_idx, matched_pred_idx = match_pointclouds(true_xyz, pred_xyz, max_distance=self.point_match_radius)
        num_unmatched_points = true_xyz.size(0) - matched_true_idx.numel()
        spatial_error = self.point_match_radius * 10 * num_unmatched_points
        prob_error = float(num_unmatched_points)
        if matched_true_xyz.numel() != 0:
            matched_distances = torch.norm(matched_true_xyz - matched_pred_xyz, dim=-1)
            spatial_error += matched_distances.mean()
            prob_error += F.mse_loss(true_probs[matched_true_idx], pred_probs[matched_pred_idx])
        loss = torch.tensor(spatial_error + prob_error, device=pred_cloud.device, requires_grad=True)
        print(f'True points {true_xyz.size(0)}, matched {matched_true_idx.numel()}, spatial_error {spatial_error}, prob_error {prob_error}, loss {loss.item()}')
        return loss


def test_match_pointclouds():
    # Test 1: No true points
    true_xyz = torch.empty((0, 3))
    pred_xyz = torch.tensor([[1.0, 2.0, 3.0]])
    matched_true_xyz, matched_pred_xyz, matched_true_idx, matched_pred_idx = match_pointclouds(true_xyz, pred_xyz)
    assert matched_true_xyz.size(0) == 0
    assert matched_pred_xyz.size(0) == 0
    assert matched_true_idx.size(0) == 0
    assert matched_pred_idx.size(0) == 0

    # Test 2: No predicted points
    true_xyz = torch.tensor([[1.0, 2.0, 3.0]])
    pred_xyz = torch.empty((0, 3))
    matched_true_xyz, matched_pred_xyz, matched_true_idx, matched_pred_idx = match_pointclouds(true_xyz, pred_xyz)
    assert matched_true_xyz.size(0) == 0
    assert matched_pred_xyz.size(0) == 0
    assert matched_true_idx.size(0) == 0
    assert matched_pred_idx.size(0) == 0

    # Test 3: No points within max_distance
    true_xyz = torch.tensor([[1.0, 1.0, 1.0]])
    pred_xyz = torch.tensor([[10.0, 10.0, 10.0]])
    max_distance = 5.0
    matched_true_xyz, matched_pred_xyz, matched_true_idx, matched_pred_idx = match_pointclouds(true_xyz, pred_xyz, max_distance)
    assert matched_true_xyz.size(0) == 0
    assert matched_pred_xyz.size(0) == 0
    assert matched_true_idx.size(0) == 0
    assert matched_pred_idx.size(0) == 0

    # Test 4: Single match
    true_xyz = torch.tensor([[1.0, 2.0, 3.0]])
    pred_xyz = torch.tensor([[1.1, 2.1, 3.1]])
    max_distance = 0.5
    matched_true_xyz, matched_pred_xyz, matched_true_idx, matched_pred_idx = match_pointclouds(true_xyz, pred_xyz, max_distance)
    assert torch.allclose(matched_true_xyz, true_xyz)
    assert torch.allclose(matched_pred_xyz, pred_xyz)
    assert torch.equal(matched_true_idx, torch.tensor([0]))
    assert torch.equal(matched_pred_idx, torch.tensor([0]))

    # Test 5: Multiple matches
    true_xyz = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    pred_xyz = torch.tensor([[1.1, 1.1, 1.1], [2.1, 2.1, 2.1]])
    max_distance = 0.5
    matched_true_xyz, matched_pred_xyz, matched_true_idx, matched_pred_idx = match_pointclouds(true_xyz, pred_xyz, max_distance)
    assert torch.allclose(matched_true_xyz, true_xyz)
    assert torch.allclose(matched_pred_xyz, pred_xyz)
    assert torch.equal(matched_true_idx, torch.tensor([0, 1]))
    assert torch.equal(matched_pred_idx, torch.tensor([0, 1]))

    # Test 6: Ambiguous matches (ensure 1-to-1 matching)
    true_xyz = torch.tensor([[1.0, 1.0, 1.0], [1.01, 1.01, 1.01], [1.2, 1.2, 1.2]])
    pred_xyz = torch.tensor([[1.1, 1.1, 1.1]])
    max_distance = 0.5
    matched_true_xyz, matched_pred_xyz, matched_true_idx, matched_pred_idx = match_pointclouds(true_xyz, pred_xyz, max_distance)
    assert matched_true_xyz.size(0) == 1
    assert matched_pred_xyz.size(0) == 1
    assert matched_true_idx.size(0) == 1
    assert matched_pred_idx.size(0) == 1
    assert torch.allclose(matched_true_xyz, torch.tensor([[1.0, 1.0, 1.0]]))
    assert torch.allclose(matched_pred_xyz, torch.tensor([[1.1, 1.1, 1.1]]))
    print('matched_true_xyz, matched_pred_xyz, matched_true_idx, matched_pred_idx')
    print(matched_true_xyz, matched_pred_xyz, matched_true_idx, matched_pred_idx)

    print("All tests passed.")


if __name__ == '__main__':
    test_match_pointclouds()
