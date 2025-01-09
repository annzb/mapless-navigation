import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftMatchingLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, smoothness_weight=0.1):
        """
        Initializes the loss function.

        Args:
            alpha (float): Weight for the spatial loss term.
            beta (float): Weight for the probability loss term.
            smoothness_weight (float): Weight for the spatial smoothness regularization.
        """
        super(SoftMatchingLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smoothness_weight = smoothness_weight

    def forward(self, predicted_points, ground_truth_points):
        """
        Computes the loss.

        Args:
            predicted_points (torch.Tensor): Predicted point cloud with shape [N, 4] ([x, y, z, prob]).
            ground_truth_points (list of torch.Tensor): List of ground truth point clouds of varying sizes [M_i, 4].

        Returns:
            torch.Tensor: Combined loss value.
        """
        pred_coords = predicted_points[:, :3]  # [N, 3]
        pred_probs = predicted_points[:, 3]  # [N]
        gt_coords = ground_truth_points[:, :3]  # [M, 3]
        gt_probs = ground_truth_points[:, 3]  # [M]

        # Compute pairwise distances
        pairwise_distances = torch.cdist(gt_coords.unsqueeze(0), pred_coords.unsqueeze(0), p=2).squeeze(0)  # [M, N]

        # Compute soft matching weights
        matching_weights = torch.softmax(-pairwise_distances, dim=-1)  # [M, N]

        # Spatial loss: Weighted squared distances
        spatial_loss = torch.sum(matching_weights * pairwise_distances.pow(2), dim=-1).mean()

        # Probability loss: Weighted mean squared error
        pred_probs_expanded = pred_probs.unsqueeze(0).expand_as(matching_weights)  # [M, N]
        gt_probs_expanded = gt_probs.unsqueeze(1).expand_as(matching_weights)  # [M, N]
        probability_loss = torch.sum(matching_weights * (pred_probs_expanded - gt_probs_expanded).pow(2), dim=-1).mean()

        # Spatial smoothness regularization
        smoothness_loss = self._compute_spatial_smoothness(pred_coords, pred_probs)

        # print(f'\nSample stats: pred points {len(pred_probs)}, true points {len(gt_probs)}, pairwise_distances.shape {pairwise_distances.shape}, matching_weights.shape {matching_weights.shape}, spatial_loss {spatial_loss}')
        # print(f'pred_probs_expanded.shape {pred_probs_expanded.shape}, gt_probs_expanded.shape {gt_probs_expanded.shape}, probability_loss {probability_loss}, smoothness_loss {smoothness_loss}')

        # Combine losses
        total_loss = self.alpha * spatial_loss + self.beta * probability_loss + self.smoothness_weight * smoothness_loss
        return total_loss

    @staticmethod
    def _compute_spatial_smoothness(coords, probs):
        """
        Computes spatial smoothness loss to encourage similar probabilities for neighboring points.

        Args:
            coords (torch.Tensor): Coordinates of shape [N, 3].
            probs (torch.Tensor): Probabilities of shape [N].

        Returns:
            torch.Tensor: Smoothness loss value.
        """
        # Compute pairwise distances
        pairwise_distances = torch.cdist(coords.unsqueeze(0), coords.unsqueeze(0), p=2).squeeze(0)  # [N, N]
        weights = torch.exp(-pairwise_distances)  # Closer points have higher weights

        # Compute pairwise differences in probabilities
        probs_diff = probs.unsqueeze(1) - probs.unsqueeze(0)  # [N, N]
        smoothness_loss = (weights * probs_diff.pow(2)).mean()  # Weighted squared differences

        return smoothness_loss



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
        # print(f'True points {true_xyz.size(0)}, matched {matched_true_idx.numel()}, spatial_error {spatial_error}, prob_error {prob_error}, loss {loss.item()}')
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


"""
1) 0 gt points?
2) bad point matching
3) loss shouldn't depend on the number of true points (need more averaging)

True points 2630, matched 21, spatial_error 26090.669921875, prob_error 2609.046630859375, loss 28699.716796875
True points 1725, matched 4, spatial_error 17210.92578125, prob_error 1721.0487060546875, loss 18931.974609375
True points 2619, matched 21, spatial_error 25980.6796875, prob_error 2598.048828125, loss 28578.728515625
True points 1907, matched 0, spatial_error 19070.0, prob_error 1907.0, loss 20977.0
True points 2539, matched 0, spatial_error 25390.0, prob_error 2539.0, loss 27929.0
True points 79, matched 0, spatial_error 790.0, prob_error 79.0, loss 869.0
True points 2305, matched 0, spatial_error 23050.0, prob_error 2305.0, loss 25355.0
True points 2628, matched 21, spatial_error 26070.677734375, prob_error 2607.0419921875, loss 28677.71875
True points 199, matched 0, spatial_error 1990.0, prob_error 199.0, loss 2189.0
True points 5, matched 0, spatial_error 50.0, prob_error 5.0, loss 55.0
True points 1115, matched 16, spatial_error 10990.677734375, prob_error 1099.05712890625, loss 12089.734375
True points 302, matched 0, spatial_error 3020.0, prob_error 302.0, loss 3322.0
True points 1721, matched 0, spatial_error 17210.0, prob_error 1721.0, loss 18931.0
True points 2658, matched 5, spatial_error 26530.759765625, prob_error 2653.062255859375, loss 29183.822265625
True points 1864, matched 0, spatial_error 18640.0, prob_error 1864.0, loss 20504.0
True points 1948, matched 0, spatial_error 19480.0, prob_error 1948.0, loss 21428.0
True points 2621, matched 21, spatial_error 26000.677734375, prob_error 2600.049072265625, loss 28600.7265625
True points 24, matched 0, spatial_error 240.0, prob_error 24.0, loss 264.0
True points 2628, matched 21, spatial_error 26070.68359375, prob_error 2607.0546875, loss 28677.73828125
True points 861, matched 0, spatial_error 8610.0, prob_error 861.0, loss 9471.0
True points 1303, matched 0, spatial_error 13030.0, prob_error 1303.0, loss 14333.0
True points 138, matched 0, spatial_error 1380.0, prob_error 138.0, loss 1518.0
True points 2624, matched 21, spatial_error 26030.67578125, prob_error 2603.034912109375, loss 28633.7109375
True points 1444, matched 0, spatial_error 14440.0, prob_error 1444.0, loss 15884.0
True points 1948, matched 0, spatial_error 19480.0, prob_error 1948.0, loss 21428.0
True points 1688, matched 9, spatial_error 16790.900390625, prob_error 1679.0498046875, loss 18469.94921875
True points 18, matched 0, spatial_error 180.0, prob_error 18.0, loss 198.0
True points 1774, matched 50, spatial_error 17240.6328125, prob_error 1724.07177734375, loss 18964.705078125
True points 1623, matched 0, spatial_error 16230.0, prob_error 1623.0, loss 17853.0
True points 2272, matched 12, spatial_error 22600.81640625, prob_error 2260.05615234375, loss 24860.873046875
True points 1512, matched 0, spatial_error 15120.0, prob_error 1512.0, loss 16632.0
True points 2528, matched 19, spatial_error 25090.68359375, prob_error 2509.033203125, loss 27599.716796875
True points 1522, matched 13, spatial_error 15090.873046875, prob_error 1509.0450439453125, loss 16599.9179687
"""