import os
import sys
import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.metrics.loss_points import DistanceLoss
from src.metrics.data_buffer import ChamferPointDataBuffer


MAX_DISTANCE = 1
LOSS_WEIGHTS = {
    'spatial_weight': 1.0,
    'occupancy_weight': 1.0,
    'unmatched_weight': 1.0,
    'fn_fp_weight': 1.0,
    'fn_weight': 1.0,
    'fp_weight': 1.0
}


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

            'expected_fn_fp_loss': 0.0,
            'expected_fn_loss': 0.0,
            'expected_fp_loss': 0.0,

            'expected_fn_fp_loss_occupied': 0.0,
            'expected_fn_loss_occupied': 0.0,
            'expected_fp_loss_occupied': 0.0
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

            'expected_fn_fp_loss': 20.0,
            'expected_fn_loss': 0.0,
            'expected_fp_loss': 0.0,

            'expected_fn_fp_loss_occupied': 20.0,
            'expected_fn_loss_occupied': 0.0,
            'expected_fp_loss_occupied': 0.0
        },
        'no_occupied_points': {
            'points1': torch.tensor([
                [0.0, 0.0, 0.0, 0.1]
            ], device=device, requires_grad=True),
            'points2': torch.tensor([
                [10.0, 0.0, 0.0, 0.3]
            ], device=device),
            'batch_size': 1,
            'batch_indices1': torch.tensor([0], device=device),
            'batch_indices2': torch.tensor([0], device=device),

            'expected_fn_fp_loss': 4.0,
            'expected_fn_loss': 0.0,
            'expected_fp_loss': 0.0,

            'expected_fn_fp_loss_occupied': 0.0,
            'expected_fn_loss_occupied': 0.0,
            'expected_fp_loss_occupied': 0.0
        },
        'occupied_pred_only': {
            'points1': torch.tensor([
                [0.0, 0.0, 0.0, 0.8]
            ], device=device, requires_grad=True),
            'points2': torch.tensor([
                [10.0, 0.0, 0.0, 0.3]
            ], device=device),
            'batch_size': 1,
            'batch_indices1': torch.tensor([0], device=device),
            'batch_indices2': torch.tensor([0], device=device),

            'expected_fn_fp_loss': 11.0,
            'expected_fn_loss': 0.0,
            'expected_fp_loss': 0.0,

            'expected_fn_fp_loss_occupied': 0.0,
            'expected_fn_loss_occupied': 0.0,
            'expected_fp_loss_occupied': 11.0
        },
        'occupied_true_only': {
            'points1': torch.tensor([
                [0.0, 0.0, 0.0, 0.3]
            ], device=device, requires_grad=True),
            'points2': torch.tensor([
                [10.0, 0.0, 0.0, 0.8]
            ], device=device),
            'batch_size': 1,
            'batch_indices1': torch.tensor([0], device=device),
            'batch_indices2': torch.tensor([0], device=device),

            'expected_fn_fp_loss': 11.0,
            'expected_fn_loss': 0.0,
            'expected_fp_loss': 0.0,

            'expected_fn_fp_loss_occupied': 0.0,
            'expected_fn_loss_occupied': 11.0,
            'expected_fp_loss_occupied': 0.0
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

            'expected_fn_fp_loss': 7.9386,
            'expected_fn_loss': 0.0,
            'expected_fp_loss': 0.0,

            'expected_fn_fp_loss_occupied': 7.9386,
            'expected_fn_loss_occupied': 0.0,
            'expected_fp_loss_occupied': 0.0
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

            'expected_fn_fp_loss': 8.8335,
            'expected_fn_loss': 0.0,
            'expected_fp_loss': 0.0,

            'expected_fn_fp_loss_occupied': 8.8335,
            'expected_fn_loss_occupied': 0.0,
            'expected_fp_loss_occupied': 0.0
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

            'expected_fn_fp_loss': 8.9201,
            'expected_fn_loss': 0.0,
            'expected_fp_loss': 0.0,

            'expected_fn_fp_loss_occupied': 8.9201,
            'expected_fn_loss_occupied': 0.0,
            'expected_fp_loss_occupied': 0.0
        }
    }

@pytest.mark.parametrize("match_occupied_only", [False, True])
@pytest.mark.parametrize("test_case", [
    'identical',
    'no_matches',
    'no_occupied_points',
    'occupied_pred_only',
    'occupied_true_only',
    'simple_match',
    'unmatched_only_pred',
    'unmatched_only_true'
])
def test_loss(device, example_clouds, match_occupied_only, test_case):
    """Test loss calculation for all example point clouds."""
    example = example_clouds[test_case]
    
    # Create buffer and loss
    buffer = ChamferPointDataBuffer(occupancy_threshold=0.5, occupied_only=match_occupied_only, max_point_distance=MAX_DISTANCE)
    buffer.create_masks(
        (example['points1'], example['batch_indices1']),
        (example['points2'], example['batch_indices2'])
    )
    loss_fn = DistanceLoss(batch_size=example['batch_size'], device=device, max_point_distance=MAX_DISTANCE, occupied_only=match_occupied_only, **LOSS_WEIGHTS)
    loss_fn.to(device)
    
    losses, loss_types = loss_fn._calc(
        (example['points1'], example['batch_indices1']),
        (example['points2'], example['batch_indices2']),
        buffer, verbose_return=True
    )
    loss = losses.mean()
    fn_fp_loss = 0 if 1 not in loss_types else losses[loss_types == 1].mean().item()
    fn_loss = 0 if 2 not in loss_types else losses[loss_types == 2].mean().item()
    fp_loss = 0 if 3 not in loss_types else losses[loss_types == 3].mean().item()
    
    # Get expected values based on match_occupied_only
    expected_fn_fp_loss = example['expected_fn_fp_loss_occupied'] if match_occupied_only else example['expected_fn_fp_loss']
    expected_fn_loss = example['expected_fn_loss_occupied'] if match_occupied_only else example['expected_fn_loss']
    expected_fp_loss = example['expected_fp_loss_occupied'] if match_occupied_only else example['expected_fp_loss']
    
    # Check losses
    assert abs(fn_fp_loss - expected_fn_fp_loss) < 1e-4, f"Failed on {test_case}"
    assert abs(fn_loss - expected_fn_loss) < 1e-4, f"Failed on {test_case}"
    assert abs(fp_loss - expected_fp_loss) < 1e-4, f"Failed on {test_case}"
    
    # Check gradient flow
    total_loss = fn_fp_loss + fn_loss + fp_loss
    assert abs(total_loss - loss.item()) < 1e-4, f"Failed on {test_case}"
    loss.backward()
    assert example['points1'].grad is not None
    assert not example['points2'].requires_grad
    