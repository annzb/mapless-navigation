import torch

from metrics.base import PointcloudOccupancyMetric


class IoU(PointcloudOccupancyMetric):
    def calc(self, predicted_points, ground_truth_points):
        """
        Computes the Intersection-over-Union (IoU) between two point clouds for batches.

        Args:
            predicted_points (list of Tensors): List of B predicted point clouds [(N1, 4), (N2, 4), ...].
            ground_truth_points (list of Tensors): List of B ground truth point clouds [(M1, 4), (M2, 4), ...].

        Returns:
            float: Average IoU over the batch.
        """
        batch_iou_scores = []
        for pred, gt in zip(predicted_points, ground_truth_points):
            pred_coords = pred[:, :3]
            gt_coords = gt[:, :3]
            if pred_coords.shape[0] == 0 or gt_coords.shape[0] == 0:
                batch_iou_scores.append(torch.tensor(0.0, device=pred_coords.device))
                continue
            pairwise_distances = torch.cdist(pred_coords.unsqueeze(0), gt_coords.unsqueeze(0), p=2).squeeze(0)
            intersection = (
                    (pairwise_distances.min(dim=1).values <= self.max_point_distance).sum() +
                    (pairwise_distances.min(dim=0).values <= self.max_point_distance).sum()
            ) // 2
            union = pred_coords.shape[0] + gt_coords.shape[0] - intersection
            iou_score = intersection / union if union > 0 else torch.tensor(0.0, device=pred_coords.device)
            batch_iou_scores.append(iou_score)
        return torch.stack(batch_iou_scores).mean().item()


class WeightedChamfer(PointcloudOccupancyMetric):
    def __call__(self, predicted_points, ground_truth_points):
        """
        Computes the weighted Chamfer distance between batches of point clouds.

        Args:
            predicted_points (list of Tensors): List of predicted point clouds [(N1, 4), (N2, 4), ...].
            ground_truth_points (list of Tensors): List of ground truth point clouds [(M1, 4), (M2, 4), ...].

        Returns:
            float: Average Weighted Chamfer distance over the batch.
        """
        batch_chamfer_distances = []

        for pred, gt in zip(predicted_points, ground_truth_points):
            pred_coords = pred[:, :3]  # [Ni, 3]
            pred_probs = pred[:, 3]  # [Ni]
            gt_coords = gt[:, :3]  # [Mi, 3]
            gt_probs = gt[:, 3]  # [Mi]

            if pred_coords.shape[0] == 0 or gt_coords.shape[0] == 0:
                batch_chamfer_distances.append(torch.tensor(0.0, device=pred_coords.device))
                continue

            pairwise_distances = torch.cdist(pred_coords.unsqueeze(0), gt_coords.unsqueeze(0), p=2).squeeze(0)

            # Forward Chamfer Distance (Predicted -> Ground Truth)
            forward_distances, forward_indices = pairwise_distances.min(dim=1)  # [Ni]
            if forward_distances.size(0) != pred_probs.size(0):
                raise ValueError(f"Mismatch in shapes: forward_distances {forward_distances.size()}, pred_probs {pred_probs.size()}")
            forward_chamfer = torch.sum(pred_probs * forward_distances ** 2) / torch.sum(pred_probs)

            # Backward Chamfer Distance (Ground Truth -> Predicted)
            backward_distances, backward_indices = pairwise_distances.min(dim=0)  # [Mi]
            if backward_distances.size(0) != gt_probs.size(0):
                raise ValueError(f"Mismatch in shapes: backward_distances {backward_distances.size()}, gt_probs {gt_probs.size()}")
            backward_chamfer = torch.sum(gt_probs * backward_distances ** 2) / torch.sum(gt_probs)

            total_chamfer = backward_chamfer + forward_chamfer
            batch_chamfer_distances.append(total_chamfer)
        return torch.stack(batch_chamfer_distances).mean().item()
