import os
import sys
import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.metrics.loss_points import PointLoss
from src.metrics.data_buffer import ChamferPointDataBuffer


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
            ], device=device),  # No requires_grad for ground truth
            'batch_size': 1,
            'batch_indices1': torch.tensor([0, 0, 0], device=device),
            'batch_indices2': torch.tensor([0, 0, 0], device=device),
            'expected_spatial_loss': 0.1732,  # sqrt(0.03) ≈ 0.1732
            'expected_prob_loss': 0.0,  # Perfect match
            'expected_unmatched_loss': 0.0,  # All points matched
            'expected_spatial_loss_occupied': 0.1732,  # Same as above since all occupied points match
            'expected_prob_loss_occupied': 0.0,  # Same as above
            'expected_unmatched_loss_occupied': 0.0,  # Same as above
        },
        'no_matches': {
            'points1': torch.tensor([
                [0.0, 0.0, 0.0, 0.7],  # Occupied
                [1.0, 1.0, 1.0, 0.3],  # Unoccupied
            ], device=device, requires_grad=True),
            'points2': torch.tensor([
                [20.0, 20.0, 20.0, 0.7],  # Far away, no matches
                [21.0, 21.0, 21.0, 0.3],  # Unoccupied
            ], device=device),  # No requires_grad for ground truth
            'batch_size': 1,
            'batch_indices1': torch.tensor([0, 0], device=device),
            'batch_indices2': torch.tensor([0, 0], device=device),
            'expected_spatial_loss': 0.0,
            'expected_prob_loss': 0.0,
            'expected_unmatched_loss': 40.0,  # 2 points * max_distance * 2 (both pred and true)
            'expected_spatial_loss_occupied': 0.0,
            'expected_prob_loss_occupied': 0.0,
            'expected_unmatched_loss_occupied': 20.0  # 1 occupied point * max_distance * 2
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
            ], device=device),  # No requires_grad for ground truth
            'batch_size': 2,
            'batch_indices1': torch.tensor([0, 0, 1], device=device),
            'batch_indices2': torch.tensor([0, 0, 1], device=device),
            'expected_spatial_loss': 0.0866,
            'expected_prob_loss': 0.0,
            'expected_unmatched_loss': 10.0,
            'expected_spatial_loss_occupied': 0.0866,
            'expected_prob_loss_occupied': 0,
            'expected_unmatched_loss_occupied': 10.0  # Same as above since both unmatched points are occupied
        },
        'different_sizes': {
            'points1': torch.tensor([
                [0.0, 0.0, 0.0, 0.7],  # Occupied
                [1.0, 1.0, 1.0, 0.3],  # Unoccupied
                [2.0, 2.0, 2.0, 0.7],  # Occupied
                [3.0, 3.0, 3.0, 0.3],  # Unoccupied
            ], device=device, requires_grad=True),
            'points2': torch.tensor([
                [0.1, 0.1, 0.1, 0.7],  # Occupied
                [1.1, 1.1, 1.1, 0.3],  # Unoccupied
                [2.1, 2.1, 2.1, 0.7],  # Occupied
            ], device=device),  # No requires_grad for ground truth
            'batch_size': 1,
            'batch_indices1': torch.tensor([0, 0, 0, 0], device=device),
            'batch_indices2': torch.tensor([0, 0, 0], device=device),
            'expected_spatial_loss': 0.1732,
            'expected_prob_loss': 0.0,
            'expected_unmatched_loss': 10.0,
            'expected_spatial_loss_occupied': 0.1732,   
            'expected_prob_loss_occupied': 0.0, 
            'expected_unmatched_loss_occupied': 0.0, 
        },
        'different_batches': {
            'points1': torch.tensor([
                [0.0, 0.0, 0.0, 0.7],  # Batch 0
                [1.0, 1.0, 1.0, 0.3],  # Batch 0
                [2.0, 2.0, 2.0, 0.7],  # Batch 1
            ], device=device, requires_grad=True),
            'points2': torch.tensor([
                [0.1, 0.1, 0.1, 0.7],  # Batch 0
                [1.1, 1.1, 1.1, 0.3],  # Batch 0
                [2.1, 2.1, 2.1, 0.7],  # Batch 2
            ], device=device),  # No requires_grad for ground truth
            'batch_size': 3,
            'batch_indices1': torch.tensor([0, 0, 1], device=device),
            'batch_indices2': torch.tensor([0, 0, 2], device=device),
            'expected_spatial_loss': 0.0577,
            'expected_prob_loss': 0.0,
            'expected_unmatched_loss': 6.6667,
            'expected_spatial_loss_occupied': 0.0577,
            'expected_prob_loss_occupied': 0.0,
            'expected_unmatched_loss_occupied': 6.6667
        },
        'multiple_unmatched': {
            'points1': torch.tensor([
                [0.0, 0.0, 0.0, 0.7],  # Matched
                [1.0, 1.0, 1.0, 0.7],  # Unmatched
                [2.0, 2.0, 2.0, 0.7],  # Unmatched
                [3.0, 3.0, 3.0, 0.3],  # Unmatched
            ], device=device, requires_grad=True),
            'points2': torch.tensor([
                [0.1, 0.1, 0.1, 0.7],  # Matched
                [10.0, 10.0, 10.0, 0.7],  # Unmatched
                [11.0, 11.0, 11.0, 0.7],  # Unmatched
            ], device=device),  # No requires_grad for ground truth
            'batch_size': 1,
            'batch_indices1': torch.tensor([0, 0, 0, 0], device=device),
            'batch_indices2': torch.tensor([0, 0, 0], device=device),
            'expected_spatial_loss': 0.1732,  # Only matched points contribute
            'expected_prob_loss': 0.0,  # Perfect match for matched points
            'expected_unmatched_loss': 50.0,
            'expected_spatial_loss_occupied': 0.1732,  # Same as above
            'expected_prob_loss_occupied': 0.0,  # Same as above
            'expected_unmatched_loss_occupied': 40.0
        },
        'mixed_occupancy': {
            'points1': torch.tensor([
                [0.0, 0.0, 0.0, 0.7],  # Matched occupied
                [1.0, 1.0, 1.0, 0.3],  # Matched unoccupied
                [2.0, 2.0, 2.0, 0.7],  # Unmatched occupied
                [3.0, 3.0, 3.0, 0.3],  # Unmatched unoccupied
            ], device=device, requires_grad=True),
            'points2': torch.tensor([
                [0.1, 0.1, 0.1, 0.7],  # Matched occupied
                [1.1, 1.1, 1.1, 0.3],  # Matched unoccupied
                [10.0, 10.0, 10.0, 0.7],  # Unmatched occupied
                [11.0, 11.0, 11.0, 0.3],  # Unmatched unoccupied
            ], device=device),  # No requires_grad for ground truth
            'batch_size': 1,
            'batch_indices1': torch.tensor([0, 0, 0, 0], device=device),
            'batch_indices2': torch.tensor([0, 0, 0, 0], device=device),
            'expected_spatial_loss': 0.1732,  # Only matched points contribute
            'expected_prob_loss': 0.0,  # Perfect match for matched points
            'expected_unmatched_loss': 40.0,
            'expected_spatial_loss_occupied': 0.1732,  # Same as above
            'expected_prob_loss_occupied': 0.0,  # Same as above
            'expected_unmatched_loss_occupied': 20.0
        },
        'empty_batches': {
            'points1': torch.tensor([
                [0.0, 0.0, 0.0, 0.7],  # Batch 0
                [1.0, 1.0, 1.0, 0.7],  # Batch 0
                [2.0, 2.0, 2.0, 0.7],  # Batch 1
            ], device=device, requires_grad=True),
            'points2': torch.tensor([
                [0.1, 0.1, 0.1, 0.7],  # Batch 0
                [1.1, 1.1, 1.1, 0.7],  # Batch 0
                [2.1, 2.1, 2.1, 0.7],  # Batch 2
            ], device=device),  # No requires_grad for ground truth
            'batch_size': 3,
            'batch_indices1': torch.tensor([0, 0, 1], device=device),
            'batch_indices2': torch.tensor([0, 0, 2], device=device),
            'expected_spatial_loss': 0.0577,  # Average over all batches
            'expected_prob_loss': 0.0,  # Perfect match for matched points
            'expected_unmatched_loss': 6.6667,  # (1 pred + 1 true) * max_distance / 3 batches
            'expected_spatial_loss_occupied': 0.0577,  # Same as above
            'expected_prob_loss_occupied': 0.0,  # Same as above
            'expected_unmatched_loss_occupied': 6.6667  # Same as above
        }
    }


@pytest.mark.parametrize("match_occupied_only", [False, True])
@pytest.mark.parametrize("test_case", [
    'simple_match',
    'no_matches',
    'mixed_batches',
    'different_sizes',
    'different_batches',
    'multiple_unmatched',
    'mixed_occupancy',
    'empty_batches'
])
def test_loss(device, example_clouds, match_occupied_only, test_case):
    """Test loss calculation for all example point clouds."""
    example = example_clouds[test_case]
    
    # Create buffer and loss
    buffer = ChamferPointDataBuffer(occupancy_threshold=0.5, match_occupied_only=match_occupied_only)
    buffer.create_masks(
        (example['points1'], example['batch_indices1']),
        (example['points2'], example['batch_indices2'])
    )
    loss_fn = PointLoss(batch_size=example['batch_size'], device=device)
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
    