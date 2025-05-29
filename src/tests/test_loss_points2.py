import os
import sys
import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.metrics.loss_points import PointLoss2
from src.metrics.data_buffer import ChamferPointDataBuffer


MAX_DISTANCE = 1


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def example_clouds(device):
    return {
        'identical': {
            'points1': torch.tensor([
                [0.0, 0.0, 0.0, 1],
                [10.0, 10.0, 10.0, 1]
            ], device=device, requires_grad=True),
            'points2': torch.tensor([
                [0.0, 0.0, 0.0, 1],
                [10.0, 10.0, 10.0, 1]
            ], device=device),
            'batch_size': 1,
            'batch_indices1': torch.tensor([0, 0], device=device),
            'batch_indices2': torch.tensor([0, 0], device=device),
            'expected_spatial_loss': 0.0, 
            'expected_prob_loss': 0.0,  # Perfect match
            'expected_unmatched_loss': 0.0,  # All points matched
            'expected_spatial_loss_occupied': 0.0,  # Same as above since all occupied points match
            'expected_prob_loss_occupied': 0.0,  # Same as above
            'expected_unmatched_loss_occupied': 0.0,  # Same as above
        },
        'no_matches': {
            'points1': torch.tensor([
                [0.0, 0.0, 0.0, 1]
            ], device=device, requires_grad=True),
            'points2': torch.tensor([
                [10.0, 0.0, 0.0, 1]
            ], device=device),
            'batch_size': 1,
            'batch_indices1': torch.tensor([0], device=device),
            'batch_indices2': torch.tensor([0], device=device),
            'expected_spatial_loss': 0.0,  # sqrt(0.03) ≈ 0.1732
            'expected_prob_loss': 0.0,  # Perfect match
            'expected_unmatched_loss': 10.0, 
            'expected_spatial_loss_occupied': 0.0,  # Same as above since all occupied points match
            'expected_prob_loss_occupied': 0.0,  # Same as above
            'expected_unmatched_loss_occupied': 10.0,  # Same as above
        },
        'simple_match': {
            'points1': torch.tensor([
                [0.0, 0.0, 0.0, 1], 
                [10.0, 10.0, 10.0, 1], 
            ], device=device, requires_grad=True),
            'points2': torch.tensor([
                [0.1, 0.1, 0.1, 1],  
                [1.0, 1.0, 1.0, 1],  
                [15.0, 15.0, 15.0, 1], 
            ], device=device),
            'batch_size': 1,
            'batch_indices1': torch.tensor([0, 0], device=device),
            'batch_indices2': torch.tensor([0, 0, 0], device=device),
            'expected_spatial_loss': 0.1732,
            'expected_prob_loss': 0.0,
            'expected_unmatched_loss': 7.2746,
            'expected_spatial_loss_occupied': 0.1732,
            'expected_prob_loss_occupied': 0.0,
            'expected_unmatched_loss_occupied': 7.2746,
        },
        'unmatched_only_pred': {
            'points1': torch.tensor([
                [0.0, 0.0, 0.0, 1], 
                [10.0, 10.0, 10.0, 1], 
            ], device=device, requires_grad=True),
            'points2': torch.tensor([
                [0.1, 0.1, 0.1, 1]
            ], device=device),
            'batch_size': 1,
            'batch_indices1': torch.tensor([0, 0], device=device),
            'batch_indices2': torch.tensor([0], device=device),
            'expected_spatial_loss': 0.1732, 
            'expected_prob_loss': 0.0,
            'expected_unmatched_loss': 5.7158,
            'expected_spatial_loss_occupied': 0.1732,
            'expected_prob_loss_occupied': 0.0,
            'expected_unmatched_loss_occupied': 5.7158,
        },
        'unmatched_only_true': {
            'points1': torch.tensor([
                [0.0, 0.0, 0.0, 1]
            ], device=device, requires_grad=True),
            'points2': torch.tensor([
                [0.1, 0.1, 0.1, 1],
                [10.0, 10.0, 10.0, 1]
            ], device=device),
            'batch_size': 1,
            'batch_indices1': torch.tensor([0], device=device),
            'batch_indices2': torch.tensor([0, 0], device=device),
            'expected_spatial_loss': 0.1732,
            'expected_prob_loss': 0.0,
            'expected_unmatched_loss': 5.7735,
            'expected_spatial_loss_occupied': 0.1732,
            'expected_prob_loss_occupied': 0.0,
            'expected_unmatched_loss_occupied': 5.7735,
        }
    }

@pytest.mark.parametrize("match_occupied_only", [False, True])
@pytest.mark.parametrize("test_case", [
    'identical',
    'simple_match',
    'no_matches',
    'unmatched_only_pred',
    'unmatched_only_true'
])
def test_loss(device, example_clouds, match_occupied_only, test_case):
    """Test loss calculation for all example point clouds."""
    example = example_clouds[test_case]
    
    # Create buffer and loss
    buffer = ChamferPointDataBuffer(occupancy_threshold=0.5, match_occupied_only=match_occupied_only, max_point_distance=MAX_DISTANCE)
    buffer.create_masks(
        (example['points1'], example['batch_indices1']),
        (example['points2'], example['batch_indices2'])
    )
    loss_fn = PointLoss2(batch_size=example['batch_size'], device=device, max_distance=MAX_DISTANCE, occupied_only=match_occupied_only)
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
    
    # Get expected values based on match_occupied_only
    expected_spatial = example['expected_spatial_loss_occupied'] if match_occupied_only else example['expected_spatial_loss']
    expected_prob = example['expected_prob_loss_occupied'] if match_occupied_only else example['expected_prob_loss']
    expected_unmatched = example['expected_unmatched_loss_occupied'] if match_occupied_only else example['expected_unmatched_loss']
    
    # Check losses
    assert abs(spatial_loss.item() - expected_spatial) < 1e-4, f"Failed on {test_case}"
    assert abs(probability_loss.item() - expected_prob) < 1e-4, f"Failed on {test_case}"
    assert abs(unmatched_loss.item() - expected_unmatched) < 1e-4, f"Failed on {test_case}"
    
    # Check gradient flow
    total_loss = spatial_loss + probability_loss + unmatched_loss
    total_loss.backward()
    assert example['points1'].grad is not None
    # Ground truth should not have gradients
    assert not example['points2'].requires_grad
    