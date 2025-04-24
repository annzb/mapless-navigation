from abc import ABC, abstractmethod
import torch


class OccupancyDataBuffer(ABC):
    def __init__(self, occupancy_threshold=0.5, **kwargs):
        self._occupancy_threshold = occupancy_threshold
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._occupied_mask = None

    def occupancy_threshold(self) -> float:
        return self._occupancy_threshold

    def occupied_mask(self):
        """
        :returns: None or a Tensor of bool.
        Size: N_points of the original point cloud.
        """
        return self._occupied_mask

    @abstractmethod
    def filter_occupied(self, y, **kwargs):
        raise NotImplementedError()

    def create_masks(self, y, **kwargs):
        self._occupied_mask = self.filter_occupied(y, **kwargs)


class PointOccupancyDataBuffer(OccupancyDataBuffer):
    def filter_occupied(self, y, **kwargs):
        cloud_values, _ = y
        mask = cloud_values[:, -1] >= self._occupancy_threshold
        return mask


class MappedPointOccupancyDataBuffer(PointOccupancyDataBuffer):
    """Buffer for handling mapped point clouds with occupancy information.
    
    This buffer extends PointOccupancyDataBuffer to handle pairs of point clouds
    and their mappings. It provides functionality for:
    - Filtering occupied points in both clouds
    - Computing and storing point mappings between clouds
    - Managing masks for mapped and occupied mapped points
    
    Attributes:
        _mapped_mask (tuple[torch.Tensor, torch.Tensor]): Pair of boolean masks indicating
            which points in each cloud are mapped to the other cloud.
        _occupied_mapped_mask (tuple[torch.Tensor, torch.Tensor]): Pair of boolean masks
            indicating which occupied points in each cloud are mapped to occupied points
            in the other cloud.
        _mapping (torch.Tensor): Soft mapping weights between points (N_matches, 2).
        _occupied_mapping (torch.Tensor): Soft mapping weights between occupied points.
        _match_occupied_only (bool): Whether to only match occupied points.
    """
    
    def __init__(self, match_occupied_only: bool = False, **kwargs):
        """Initialize the buffer.
        
        Args:
            match_occupied_only: If True, only match occupied points between clouds.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self._mapped_mask = None
        self._occupied_mapped_mask = None
        self._mapping = None
        self._occupied_mapping = None
        self._match_occupied_only = match_occupied_only

    def filter_occupied(self, y, y_other=None, **kwargs):
        """Filter occupied points in both clouds.
        
        Args:
            y: Tuple of (points, batch_indices) for first cloud.
            y_other: Tuple of (points, batch_indices) for second cloud.
            **kwargs: Additional arguments.
            
        Returns:
            Tuple of (mask, mask_other) indicating occupied points in each cloud.
        """
        if y_other is None:
            raise ValueError('y_other not provided.')
        y_values, _ = y
        y_values_other, _ = y_other
        mask = y_values[:, -1] >= self._occupancy_threshold
        mask_other = y_values_other[:, -1] >= self._occupancy_threshold
        return mask, mask_other

    def create_masks(self, y, y_other=None, **kwargs):
        """Create all necessary masks for the input clouds.
        
        Args:
            y: Tuple of (points, batch_indices) for first cloud.
            y_other: Tuple of (points, batch_indices) for second cloud.
            **kwargs: Additional arguments.
        """
        super().create_masks(y, y_other=y_other, **kwargs)
        if y_other is None:
            raise ValueError('y_other not provided.')
            
        (y_values, y_batch_indices), (y_values_other, y_batch_indices_other) = y, y_other
        
        # Validate input shapes
        assert y_values.shape[0] == y_batch_indices.shape[0], "y_values and y_batch_indices size mismatch"
        assert y_values_other.shape[0] == y_batch_indices_other.shape[0], "y_values_other and y_batch_indices_other size mismatch"
        
        # Get occupied masks
        occupied_mask, occupied_mask_other = self._occupied_mask
        
        # Initialize masks
        mapped_mask = torch.zeros_like(occupied_mask, dtype=torch.bool)
        mapped_mask_other = torch.zeros_like(occupied_mask_other, dtype=torch.bool)
        
        # Compute mapping
        if self._match_occupied_only:
            # Only match occupied points
            y_values_masked = y_values[occupied_mask]
            y_values_other_masked = y_values_other[occupied_mask_other]
            y_batch_indices_masked = y_batch_indices[occupied_mask]
            y_batch_indices_other_masked = y_batch_indices_other[occupied_mask_other]
        else:
            # Match all points
            y_values_masked = y_values
            y_values_other_masked = y_values_other
            y_batch_indices_masked = y_batch_indices
            y_batch_indices_other_masked = y_batch_indices_other
            
        # Compute soft mapping
        mapping = self.match_points(
            y_values_masked, y_batch_indices_masked,
            y_values_other_masked, y_batch_indices_other_masked
        )
        
        # Store mapping
        self._mapping = mapping
        
        # Create mapped masks
        if mapping.numel() > 0:
            if self._match_occupied_only:
                # For occupied-only matching, mapped masks are the same as occupied masks
                mapped_mask = occupied_mask
                mapped_mask_other = occupied_mask_other
            else:
                # For all-point matching, create masks from mapping
                # Use only first two columns for indexing (indices) if mapping has more than 2 columns
                indices = mapping[:, :2].long() if mapping.shape[1] > 2 else mapping.long()
                mapped_mask[indices[:, 0]] = True
                mapped_mask_other[indices[:, 1]] = True
                
        self._mapped_mask = (mapped_mask, mapped_mask_other)
        
        # Create occupied mapped masks
        if self._match_occupied_only:
            # For occupied-only matching, occupied mapped masks are the same as mapped masks
            self._occupied_mapped_mask = self._mapped_mask
        else:
            # For all-point matching, combine occupied and mapped masks
            occupied_mapped_mask = torch.zeros_like(occupied_mask, dtype=torch.bool)
            occupied_mapped_mask_other = torch.zeros_like(occupied_mask_other, dtype=torch.bool)
            if mapping.numel() > 0:
                # Only set True for points that are both mapped and occupied
                indices = mapping[:, :2].long() if mapping.shape[1] > 2 else mapping.long()
                for i in range(len(indices)):
                    idx1, idx2 = indices[i]
                    if occupied_mask[idx1] and occupied_mask_other[idx2]:
                        occupied_mapped_mask[idx1] = True
                        occupied_mapped_mask_other[idx2] = True
            self._occupied_mapped_mask = (occupied_mapped_mask, occupied_mapped_mask_other)
            
        # Skip validation for empty mappings
        if mapping.numel() > 0:
            # Validate masks
            assert mapped_mask.sum() == mapped_mask_other.sum(), "Mismatch in number of mapped points"
            if not self._match_occupied_only:
                assert occupied_mapped_mask.sum() == occupied_mapped_mask_other.sum(), "Mismatch in number of occupied mapped points"

    def mapped_mask(self):
        """Get the pair of masks indicating mapped points.
        
        Returns:
            Tuple of (mask, mask_other) where each mask is a boolean tensor
            indicating which points are mapped to the other cloud.
        """
        return self._mapped_mask

    def occupied_mapped_mask(self):
        """Get the pair of masks indicating occupied mapped points.
        
        Returns:
            Tuple of (mask, mask_other) where each mask is a boolean tensor
            indicating which occupied points are mapped to occupied points
            in the other cloud.
        """
        return self._occupied_mapped_mask

    def mapping(self):
        """Get the soft mapping between points.
        
        Returns:
            Tensor of shape (N_matches, 2) containing the soft mapping weights
            between matched points.
        """
        return self._mapping

    def occupied_mapping(self):
        """Get the soft mapping between occupied points.
        
        Returns:
            Tensor of shape (N_matches, 2) containing the soft mapping weights
            between matched occupied points.
        """
        return self._occupied_mapping

    @abstractmethod
    def match_points(self, cloud_values_1, batch_indices_1, cloud_values_2, batch_indices_2, **kwargs):
        """Compute soft mapping between points in two clouds.
        
        Args:
            cloud_values_1: Points from first cloud.
            batch_indices_1: Batch indices for first cloud.
            cloud_values_2: Points from second cloud.
            batch_indices_2: Batch indices for second cloud.
            **kwargs: Additional arguments.
            
        Returns:
            Tensor of shape (N_matches, 2) containing the soft mapping weights
            between matched points.
        """
        raise NotImplementedError()


class ChamferPointDataBuffer(MappedPointOccupancyDataBuffer):
    """Buffer that uses Chamfer distance for point matching.
    
    This buffer implements hard matching between points using Chamfer distance.
    It matches each point to its nearest neighbor in the other cloud.
    """

    def match_points(self, cloud_values_1, batch_indices_1, cloud_values_2, batch_indices_2, **kwargs):
        """Match points using Chamfer distance.
        
        Args:
            cloud_values_1: Points from first cloud.
            batch_indices_1: Batch indices for first cloud.
            cloud_values_2: Points from second cloud.
            batch_indices_2: Batch indices for second cloud.
            **kwargs: Additional arguments.
            
        Returns:
            Tensor of shape (N_matches, 2) containing the indices of matched points.
        """
        # Handle empty input cases
        if cloud_values_1.size(0) == 0 or cloud_values_2.size(0) == 0:
            return torch.zeros((0, 2), device=cloud_values_1.device)
            
        # Compute pairwise distances
        dists = torch.cdist(cloud_values_1[:, :3], cloud_values_2[:, :3])
        
        # Find nearest neighbors
        best_12 = dists.argmin(dim=1)  # For each point in cloud 1, find closest in cloud 2
        best_21 = dists.argmin(dim=0)  # For each point in cloud 2, find closest in cloud 1
        
        # Find mutual matches
        idx1 = torch.arange(cloud_values_1.size(0), device=cloud_values_1.device)
        mutual_mask = (best_21[best_12] == idx1)
        
        # Get matched indices
        matched_idx1 = idx1[mutual_mask]
        matched_idx2 = best_12[mutual_mask]
        
        # Create mapping tensor
        mapping = torch.stack([matched_idx1, matched_idx2], dim=1)
        
        # Filter matches by batch and distance threshold
        valid_matches = []
        max_distance = 10.0  # Maximum distance threshold for valid matches
        for i in range(len(mapping)):
            idx1, idx2 = mapping[i]
            if batch_indices_1[idx1] == batch_indices_2[idx2]:
                # Check if points are close enough
                dist = dists[idx1, idx2]
                if dist < max_distance:
                    valid_matches.append(i)
                
        if len(valid_matches) == 0:
            return torch.zeros((0, 2), device=cloud_values_1.device)
            
        mapping = mapping[valid_matches]
        return mapping


class SinkhornPointDataBuffer(MappedPointOccupancyDataBuffer):
    """Buffer that uses Sinkhorn algorithm for soft point matching.
    
    This buffer implements soft matching between points using the Sinkhorn algorithm.
    It computes a doubly stochastic matrix that represents the soft assignment
    between points in the two clouds.
    """
    
    def __init__(self, n_iters: int = 20, temperature: float = 0.1, **kwargs):
        """Initialize the buffer.
        
        Args:
            n_iters: Number of Sinkhorn iterations.
            temperature: Temperature parameter for softmax.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self._n_iters = n_iters
        self._temperature = temperature
        self._soft_assignment = None

    def soft_assignment(self):
        """Get the soft assignment matrix.
        
        Returns:
            Tensor of shape (N1, N2) containing the soft assignment weights
            between points in the two clouds.
        """
        return self._soft_assignment

    def _sinkhorn_normalization(self, log_alpha):
        """Apply Sinkhorn normalization to the log assignment matrix.
        
        Args:
            log_alpha: Log assignment matrix.
            
        Returns:
            Normalized assignment matrix.
        """
        for _ in range(self._n_iters):
            # Row normalization
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
            # Column normalization
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=0, keepdim=True)
        return torch.exp(log_alpha)

    def match_points(self, cloud_values_1, batch_indices_1, cloud_values_2, batch_indices_2, **kwargs):
        """Match points using Sinkhorn algorithm.
        
        Args:
            cloud_values_1: Points from first cloud.
            batch_indices_1: Batch indices for first cloud.
            cloud_values_2: Points from second cloud.
            batch_indices_2: Batch indices for second cloud.
            **kwargs: Additional arguments.
            
        Returns:
            Tensor of shape (N_matches, 2) containing the indices of matched points
            and their soft assignment weights.
        """
        # Compute pairwise distances
        dists = torch.cdist(cloud_values_1[:, :3], cloud_values_2[:, :3])
        
        # Compute log assignment matrix
        log_alpha = -dists / self._temperature
        
        # Apply Sinkhorn normalization
        assignment = self._sinkhorn_normalization(log_alpha)
        self._soft_assignment = assignment
        
        # Find hard matches (for compatibility with other buffers)
        best_12 = assignment.argmax(dim=1)
        best_21 = assignment.argmax(dim=0)
        idx1 = torch.arange(cloud_values_1.size(0), device=cloud_values_1.device)
        mutual_mask = (best_21[best_12] == idx1)
        
        # Get matched indices
        matched_idx1 = idx1[mutual_mask]
        matched_idx2 = best_12[mutual_mask]
        
        # Create mapping tensor with soft weights
        mapping = torch.stack([
            matched_idx1,
            matched_idx2,
            assignment[matched_idx1, matched_idx2]
        ], dim=1)
        
        return mapping
