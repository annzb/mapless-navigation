import torch

from base_classes import PointcloudOccupancyMetric


class IoU(PointcloudOccupancyMetric):
    def calc(self, predicted_points, ground_truth_points):
        """
        Computes the Intersection-over-Union (IoU) between two point clouds.

        Args:
            predicted_points (torch.Tensor): Predicted point cloud of shape [N, 4] ([x, y, z, prob]).
            ground_truth_points (torch.Tensor): Ground truth point cloud of shape [M, 4].
            max_distance (float): Overlap radius.
            probability_threshold (float): Threshold to filter points by their probabilities (default=0.5).

        Returns:
            float: IoU value between predicted and ground truth point clouds.
        """
        pred_coords = predicted_points[:, :3]
        gt_coords = ground_truth_points[:, :3]

        # Count intersections based on closeness
        pairwise_distances = torch.cdist(pred_coords.unsqueeze(0), gt_coords.unsqueeze(0), p=2).squeeze(0)
        intersection = (
                (pairwise_distances.min(dim=1).values <= self.max_point_distance).sum() +
                (pairwise_distances.min(dim=0).values <= self.max_point_distance).sum()
        ) // 2
        union = len(pred_coords) + len(gt_coords) - intersection

        if union == 0:
            return 0.0
        iou_score = intersection / union
        return iou_score.item()


class WeightedChamfer(PointcloudOccupancyMetric):
    def __call__(self, predicted_points, ground_truth_points):
        """
        Computes the weighted Chamfer distance between two point clouds.

        Args:
            predicted_points (torch.Tensor): Predicted point cloud of shape [N, 4] ([x, y, z, prob]).
            ground_truth_points (torch.Tensor): Ground truth point cloud of shape [M, 4].

        Returns:
            float: Weighted Chamfer distance.
        """
        pred_coords = predicted_points[:, :3]
        pred_probs = predicted_points[:, 3]
        gt_coords = ground_truth_points[:, :3]
        gt_probs = ground_truth_points[:, 3]

        pairwise_distances = torch.cdist(pred_coords.unsqueeze(0), gt_coords.unsqueeze(0), p=2).squeeze(0)

        # Forward Chamfer Distance (Predicted -> Ground Truth)
        forward_distances, forward_indices = pairwise_distances.min(dim=1)  # For each predicted point
        if forward_distances.size(0) != pred_probs.size(0):
            raise ValueError(f"Mismatch in shapes: forward_distances {forward_distances.size()}, pred_probs {pred_probs.size()}")
        forward_chamfer = torch.sum(pred_probs * forward_distances ** 2) / torch.sum(pred_probs)

        # Backward Chamfer Distance (Ground Truth -> Predicted)
        backward_distances, backward_indices = pairwise_distances.min(dim=0)  # For each ground truth point
        if backward_distances.size(0) != gt_probs.size(0):
            raise ValueError(f"Mismatch in shapes: backward_distances {backward_distances.size()}, gt_probs {gt_probs.size()}")
        backward_chamfer = torch.sum(gt_probs * backward_distances ** 2) / torch.sum(gt_probs)

        # Total Chamfer Distance
        total_chamfer = backward_chamfer + forward_chamfer
        return total_chamfer.item()
