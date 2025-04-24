import os
import sys
import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.metrics.loss_points import ChamferPointLoss
from src.metrics.data_buffer import ChamferPointDataBuffer


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def example_clouds(device):
    return {
        'simple_match': {
            'points1': torch.tensor([
                [0.0, 0.0, 0.0, 0.7],  # Occupied
                [1.0, 1.0, 1.0, 0.3],  # Unoccupied
                [2.0, 2.0, 2.0, 0.7],  # Occupied
            ], device=device, requires_grad=True),
            'points2': torch.tensor([
                [0.1, 0.1, 0.1, 0.7],  # Occupied
                [1.1, 1.1, 1.1, 0.3],  # Unoccupied
                [2.1, 2.1, 2.1, 0.7],  # Occupied
            ], device=device, requires_grad=True),
            'batch_indices1': torch.tensor([0, 0, 0], device=device),
            'batch_indices2': torch.tensor([0, 0, 0], device=device),
            'expected_spatial_loss': 0.1732,  # Approximate value
            'expected_prob_loss': 0.0,  # Perfect match
            'expected_unmatched_loss': 0.0,  # All points matched
        },
        'no_matches': {
            'points1': torch.tensor([
                [0.0, 0.0, 0.0, 0.7],
                [1.0, 1.0, 1.0, 0.3],
            ], device=device, requires_grad=True),
            'points2': torch.tensor([
                [20.0, 20.0, 20.0, 0.7],  # Far away, no matches
                [21.0, 21.0, 21.0, 0.3],
            ], device=device, requires_grad=True),
            'batch_indices1': torch.tensor([0, 0], device=device),
            'batch_indices2': torch.tensor([0, 0], device=device),
            'expected_spatial_loss': 10.0,  # max_distance
            'expected_prob_loss': 1.0,  # max probability loss
            'expected_unmatched_loss': 20.0,  # 2 points * max_distance
        },
        'mixed_batches': {
            'points1': torch.tensor([
                [0.0, 0.0, 0.0, 0.7],  # Matched
                [1.0, 1.0, 1.0, 0.3],  # Matched
                [10.0, 10.0, 10.0, 0.7],  # Unmatched
            ], device=device, requires_grad=True),
            'points2': torch.tensor([
                [0.1, 0.1, 0.1, 0.7],  # Matched
                [1.1, 1.1, 1.1, 0.3],  # Matched
                [20.0, 20.0, 20.0, 0.7],  # Unmatched
            ], device=device, requires_grad=True),
            'batch_indices1': torch.tensor([0, 0, 1], device=device),
            'batch_indices2': torch.tensor([0, 0, 1], device=device),
            'expected_spatial_loss': 0.1732,  # Approximate value for matched points
            'expected_prob_loss': 0.0,  # Perfect match for matched points
            'expected_unmatched_loss': 10.0,  # 1 point * max_distance
        }
    }


def test_loss_calculation(device, batch_size, example_clouds):
    """Test loss calculation for all example point clouds."""
    for name, example in example_clouds.items():
        # Create buffer and loss
        buffer = ChamferPointDataBuffer(occupancy_threshold=0.5)
        buffer.create_masks(
            (example['points1'], example['batch_indices1']),
            (example['points2'], example['batch_indices2'])
        )
        loss_fn = ChamferPointLoss(batch_size=batch_size, device=device)
        loss_fn.to(device)
        
        # Calculate losses
        spatial_loss, probability_loss = loss_fn._calc_matched_loss(
            (example['points1'], example['batch_indices1']),
            (example['points2'], example['batch_indices2']),
            buffer
        )
        unmatched_loss = loss_fn._calc_unmatched_loss(
            (example['points1'], example['batch_indices1']),
            (example['points2'], example['batch_indices2']),
            buffer
        )
        
        # Check losses
        assert abs(spatial_loss.item() - example['expected_spatial_loss']) < 1e-4
        assert abs(probability_loss.item() - example['expected_prob_loss']) < 1e-4
        assert abs(unmatched_loss.item() - example['expected_unmatched_loss']) < 1e-4
        
        # Check gradient flow
        spatial_loss.backward()
        assert example['points1'].grad is not None
        assert example['points2'].grad is not None


def test_occupied_only(device, batch_size, example_clouds):
    """Test loss calculation with occupied_only=True."""
    for name, example in example_clouds.items():
        # Create buffer and loss with occupied_only=True
        buffer = ChamferPointDataBuffer(occupancy_threshold=0.5)
        buffer.create_masks(
            (example['points1'], example['batch_indices1']),
            (example['points2'], example['batch_indices2'])
        )
        loss_fn = ChamferPointLoss(batch_size=batch_size, device=device, occupied_only=True)
        loss_fn.to(device)
        
        # Calculate losses
        spatial_loss, probability_loss = loss_fn._calc_matched_loss(
            (example['points1'], example['batch_indices1']),
            (example['points2'], example['batch_indices2']),
            buffer
        )
        unmatched_loss = loss_fn._calc_unmatched_loss(
            (example['points1'], example['batch_indices1']),
            (example['points2'], example['batch_indices2']),
            buffer
        )
        
        # Check that losses are positive and require gradients
        assert spatial_loss.item() > 0
        assert probability_loss.item() > 0
        assert unmatched_loss.item() > 0
        assert spatial_loss.requires_grad
        assert probability_loss.requires_grad
        assert unmatched_loss.requires_grad
        
        # Check gradient flow
        spatial_loss.backward()
        assert example['points1'].grad is not None
        assert example['points2'].grad is not None
    