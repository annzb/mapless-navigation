import pytest
import torch
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from src.metrics.data_buffer import (
    OccupancyDataBuffer,
    PointOccupancyDataBuffer,
    MappedPointOccupancyDataBuffer,
    ChamferPointDataBuffer,
    SinkhornPointDataBuffer
)


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_point_occupancy_buffer():
    # Create test data
    points = torch.tensor([
        [0.0, 0.0, 0.0, 0.3],  # Below threshold
        [1.0, 1.0, 1.0, 0.7],  # Above threshold
        [2.0, 2.0, 2.0, 0.5],  # At threshold
    ])
    batch_indices = torch.tensor([0, 0, 0], dtype=torch.int)
    
    # Create buffer
    buffer = PointOccupancyDataBuffer(occupancy_threshold=0.5)
    buffer.create_masks((points, batch_indices))
    
    # Check occupied mask
    occupied_mask = buffer.occupied_mask()
    assert occupied_mask.shape == (3,)
    assert occupied_mask[0].item() == False
    assert occupied_mask[1].item() == True
    assert occupied_mask[2].item() == True


def test_basic_matching(device):
    # Create test data with clear matches
    points1 = torch.tensor([
        [0.0, 0.0, 0.0, 0.7],
        [1.0, 1.0, 1.0, 0.3],
        [2.0, 2.0, 2.0, 0.7],
    ], device=device)
    points2 = torch.tensor([
        [0.1, 0.1, 0.1, 0.7],
        [1.1, 1.1, 1.1, 0.3],
        [2.1, 2.1, 2.1, 0.7],
    ], device=device)
    batch_indices1 = torch.tensor([0, 0, 0], device=device, dtype=torch.int)
    batch_indices2 = torch.tensor([0, 0, 0], device=device, dtype=torch.int)
    
    # Create buffer
    buffer = ChamferPointDataBuffer(occupancy_threshold=0.5)
    buffer.create_masks((points1, batch_indices1), (points2, batch_indices2))

    occupied_mask, occupied_mask_other = buffer.occupied_mask()
    expected_occupied_mask = torch.tensor([True, False, True], device=device)
    expected_occupied_mask_other = torch.tensor([True, False, True], device=device)
    assert torch.equal(occupied_mask, expected_occupied_mask), f"Expected occupied_mask to be {expected_occupied_mask}, got {occupied_mask}"
    assert torch.equal(occupied_mask_other, expected_occupied_mask_other), f"Expected occupied_mask_other to be {expected_occupied_mask_other}, got {occupied_mask_other}"
    
    mapping = buffer.mapping()
    expected_mapping = torch.tensor([[0, 0], [1, 1], [2, 2]], device=device)
    assert torch.equal(mapping, expected_mapping), f"Expected mapping to be {expected_mapping}, got {mapping}"

    mapped_mask, mapped_mask_other = buffer.mapped_mask()
    expected_mapped_mask = torch.tensor([True, True, True], device=device)
    expected_mapped_mask_other = torch.tensor([True, True, True], device=device)
    assert torch.equal(mapped_mask, expected_mapped_mask), f"Expected mapped_mask to be {expected_mapped_mask}, got {mapped_mask}"
    assert torch.equal(mapped_mask_other, expected_mapped_mask_other), f"Expected mapped_mask_other to be {expected_mapped_mask_other}, got {mapped_mask_other}"

    buffer_occ_only = ChamferPointDataBuffer(occupancy_threshold=0.5, match_occupied_only=True)
    buffer_occ_only.create_masks((points1, batch_indices1), (points2, batch_indices2))

    occupied_mask, occupied_mask_other = buffer_occ_only.occupied_mask()
    assert torch.equal(occupied_mask, expected_occupied_mask), f"Expected occupied_mask to be {expected_occupied_mask}, got {occupied_mask}"
    assert torch.equal(occupied_mask_other, expected_occupied_mask_other), f"Expected occupied_mask_other to be {expected_occupied_mask_other}, got {occupied_mask_other}"

    mapping = buffer_occ_only.mapping()
    expected_mapping = torch.tensor([[0, 0], [2, 2]], device=device)
    assert torch.equal(mapping, expected_mapping), f"Expected mapping to be {expected_mapping}, got {mapping}"

    mapped_mask, mapped_mask_other = buffer_occ_only.mapped_mask()
    expected_mapped_mask = torch.tensor([True, False, True], device=device)
    expected_mapped_mask_other = torch.tensor([True, False, True], device=device)
    assert torch.equal(mapped_mask, expected_mapped_mask), f"Expected mapped_mask to be {expected_mapped_mask}, got {mapped_mask}"
    assert torch.equal(mapped_mask_other, expected_mapped_mask_other), f"Expected mapped_mask_other to be {expected_mapped_mask_other}, got {mapped_mask_other}"


def test_empty_mapping(device):
    # Create test data with no possible matches (points too far apart)
    points1 = torch.tensor([
        [0.0, 0.0, 0.0, 0.7],
        [1.0, 1.0, 1.0, 0.3],
    ], device=device)
    points2 = torch.tensor([
        [10.0, 10.0, 10.0, 0.7],
        [11.0, 11.0, 11.0, 0.3],
    ], device=device)
    batch_indices1 = torch.tensor([0, 0], device=device, dtype=torch.int)
    batch_indices2 = torch.tensor([0, 0], device=device, dtype=torch.int)
    
    # Test regular matching
    buffer = ChamferPointDataBuffer(occupancy_threshold=0.5)
    buffer.create_masks((points1, batch_indices1), (points2, batch_indices2))
    
    # Check empty mapping
    mapping = buffer.mapping()
    expected_mapping = torch.zeros((0, 2), device=device, dtype=mapping.dtype)
    assert torch.equal(mapping, expected_mapping), f"Expected mapping to be {expected_mapping}, got {mapping}"
    
    # Check empty masks
    mapped_mask, mapped_mask_other = buffer.mapped_mask()
    expected_mapped_mask = torch.tensor([False, False], device=device)
    expected_mapped_mask_other = torch.tensor([False, False], device=device)
    assert torch.equal(mapped_mask, expected_mapped_mask), f"Expected mapped_mask to be {expected_mapped_mask}, got {mapped_mask}"
    assert torch.equal(mapped_mask_other, expected_mapped_mask_other), f"Expected mapped_mask_other to be {expected_mapped_mask_other}, got {mapped_mask_other}"
    
    # Test occupied-only matching
    buffer_occ_only = ChamferPointDataBuffer(occupancy_threshold=0.5, match_occupied_only=True)
    buffer_occ_only.create_masks((points1, batch_indices1), (points2, batch_indices2))
    
    # Check empty mapping for occupied-only buffer
    mapping = buffer_occ_only.mapping()
    expected_mapping = torch.zeros((0, 2), device=device, dtype=mapping.dtype)
    assert torch.equal(mapping, expected_mapping), f"Expected mapping to be {expected_mapping}, got {mapping}"
    
    # Check empty masks for occupied-only buffer
    mapped_mask, mapped_mask_other = buffer_occ_only.mapped_mask()
    expected_mapped_mask = torch.tensor([False, False], device=device)
    expected_mapped_mask_other = torch.tensor([False, False], device=device)
    assert torch.equal(mapped_mask, expected_mapped_mask), f"Expected mapped_mask to be {expected_mapped_mask}, got {mapped_mask}"
    assert torch.equal(mapped_mask_other, expected_mapped_mask_other), f"Expected mapped_mask_other to be {expected_mapped_mask_other}, got {mapped_mask_other}"


def test_batch_aware_matching(device):
    # Create test data with multiple batches
    points1 = torch.tensor([
        [0.0, 0.0, 0.0, 0.7],  # Batch 0
        [1.0, 1.0, 1.0, 0.7],  # Batch 1
        [0.0, 0.0, 0.0, 0.6],  # Batch 1
    ], device=device)
    points2 = torch.tensor([
        [0.1, 0.1, 0.1, 0.7],  # Batch 0
        [1.1, 1.1, 1.1, 0.66],  # Batch 0
        [1.1, 1.1, 1.1, 0.7],  # Batch 1
        [0.1, 0.1, 0.1, 0.4],  # Batch 1
    ], device=device)
    batch_indices1 = torch.tensor([0, 1, 1], device=device, dtype=torch.int)
    batch_indices2 = torch.tensor([0, 0, 1, 1], device=device, dtype=torch.int)
    
    buffer = ChamferPointDataBuffer(occupancy_threshold=0.5)
    buffer.create_masks((points1, batch_indices1), (points2, batch_indices2))
    
    # Check occupied masks
    occupied_mask, occupied_mask_other = buffer.occupied_mask()
    expected_occupied_mask = torch.tensor([True, True, True], device=device)
    expected_occupied_mask_other = torch.tensor([True, True, True, False], device=device)
    assert torch.equal(occupied_mask, expected_occupied_mask), f"Expected occupied_mask to be {expected_occupied_mask}, got {occupied_mask}"
    assert torch.equal(occupied_mask_other, expected_occupied_mask_other), f"Expected occupied_mask_other to be {expected_occupied_mask_other}, got {occupied_mask_other}"
    
    # Check mapping
    mapping = buffer.mapping()
    expected_mapping = torch.tensor([[0, 0], [1, 2], [2, 3]], device=device)
    assert torch.equal(mapping, expected_mapping), f"Expected mapping to be {expected_mapping}, got {mapping}"
    
    # Check mapped masks
    mapped_mask, mapped_mask_other = buffer.mapped_mask()
    expected_mapped_mask = torch.tensor([True, True, True], device=device)
    expected_mapped_mask_other = torch.tensor([True, False, True, True], device=device)
    assert torch.equal(mapped_mask, expected_mapped_mask), f"Expected mapped_mask to be {expected_mapped_mask}, got {mapped_mask}"
    assert torch.equal(mapped_mask_other, expected_mapped_mask_other), f"Expected mapped_mask_other to be {expected_mapped_mask_other}, got {mapped_mask_other}"

    # Test with occupied-only matching
    buffer_occ_only = ChamferPointDataBuffer(occupancy_threshold=0.5, match_occupied_only=True)
    buffer_occ_only.create_masks((points1, batch_indices1), (points2, batch_indices2))
    
    # Check occupied masks
    occupied_mask, occupied_mask_other = buffer_occ_only.occupied_mask()
    expected_occupied_mask = torch.tensor([True, True, True], device=device)
    expected_occupied_mask_other = torch.tensor([True, True, True, False], device=device)
    assert torch.equal(occupied_mask, expected_occupied_mask), f"Expected occupied_mask to be {expected_occupied_mask}, got {occupied_mask}"
    assert torch.equal(occupied_mask_other, expected_occupied_mask_other), f"Expected occupied_mask_other to be {expected_occupied_mask_other}, got {occupied_mask_other}"
    
    # Check mapping - should only include occupied points
    mapping = buffer_occ_only.mapping()
    expected_mapping = torch.tensor([[0, 0], [1, 2]], device=device)
    assert torch.equal(mapping, expected_mapping), f"Expected mapping to be {expected_mapping}, got {mapping}"
    
    # Check mapped masks
    mapped_mask, mapped_mask_other = buffer_occ_only.mapped_mask()
    expected_mapped_mask = torch.tensor([True, True, False], device=device)
    expected_mapped_mask_other = torch.tensor([True, False, True, False], device=device)
    assert torch.equal(mapped_mask, expected_mapped_mask), f"Expected mapped_mask to be {expected_mapped_mask}, got {mapped_mask}"
    assert torch.equal(mapped_mask_other, expected_mapped_mask_other), f"Expected mapped_mask_other to be {expected_mapped_mask_other}, got {mapped_mask_other}"


def test_single_point_matching(device):
    # Test with single point in each cloud
    points1 = torch.tensor([[0.0, 0.0, 0.0, 0.7]], device=device)
    points2 = torch.tensor([[0.1, 0.1, 0.1, 0.7]], device=device)
    batch_indices1 = torch.tensor([0], device=device)
    batch_indices2 = torch.tensor([0], device=device)
    
    # Test regular matching
    buffer = ChamferPointDataBuffer(occupancy_threshold=0.5)
    buffer.create_masks((points1, batch_indices1), (points2, batch_indices2))
    
    # Check single point mapping
    mapping = buffer.mapping()
    expected_mapping = torch.tensor([[0, 0]], device=device)
    assert torch.equal(mapping, expected_mapping), f"Expected mapping to be {expected_mapping}, got {mapping}"
    
    # Check masks
    mapped_mask, mapped_mask_other = buffer.mapped_mask()
    expected_mapped_mask = torch.tensor([True], device=device)
    expected_mapped_mask_other = torch.tensor([True], device=device)
    assert torch.equal(mapped_mask, expected_mapped_mask), f"Expected mapped_mask to be {expected_mapped_mask}, got {mapped_mask}"
    assert torch.equal(mapped_mask_other, expected_mapped_mask_other), f"Expected mapped_mask_other to be {expected_mapped_mask_other}, got {mapped_mask_other}"
    
    # Test with unoccupied point
    points1_unocc = torch.tensor([[0.0, 0.0, 0.0, 0.3]], device=device)
    buffer_unocc = ChamferPointDataBuffer(occupancy_threshold=0.5)
    buffer_unocc.create_masks((points1_unocc, batch_indices1), (points2, batch_indices2))
    
    # Check mapping still exists
    mapping_unocc = buffer_unocc.mapping()
    expected_mapping_unocc = torch.tensor([[0, 0]], device=device)
    assert torch.equal(mapping_unocc, expected_mapping_unocc), f"Expected mapping to be {expected_mapping_unocc}, got {mapping_unocc}"

    # Test with occupied-only matching
    buffer_occ_only = ChamferPointDataBuffer(occupancy_threshold=0.5, match_occupied_only=True)
    buffer_occ_only.create_masks((points1_unocc, batch_indices1), (points2, batch_indices2))
    
    # Check no mapping exists for unoccupied point
    mapping_occ_only = buffer_occ_only.mapping()
    expected_mapping_occ_only = torch.zeros((0, 2), device=device, dtype=mapping_occ_only.dtype)
    assert torch.equal(mapping_occ_only, expected_mapping_occ_only), f"Expected mapping to be {expected_mapping_occ_only}, got {mapping_occ_only}"
    
    # Check masks
    mapped_mask, mapped_mask_other = buffer_occ_only.mapped_mask()
    expected_mapped_mask = torch.tensor([False], device=device)
    expected_mapped_mask_other = torch.tensor([False], device=device)
    assert torch.equal(mapped_mask, expected_mapped_mask), f"Expected mapped_mask to be {expected_mapped_mask}, got {mapped_mask}"
    assert torch.equal(mapped_mask_other, expected_mapped_mask_other), f"Expected mapped_mask_other to be {expected_mapped_mask_other}, got {mapped_mask_other}"


def test_mixed_occupancy_matching(device):
    # Test with mixed occupancy values and multiple batches
    points1 = torch.tensor([
        [2.0, 2.0, 2.0, 0.7],  # Batch 0, Occupied  T
        [1.0, 1.0, 1.0, 0.3],  # Batch 0, Unoccupied
        [0.0, 0.0, 0.0, 0.7],  # Batch 0, Occupied  T
        [2.0, 2.0, 2.0, 0.7],  # Batch 1, Occupied
        [3.0, 3.0, 3.0, 0.4],  # Batch 1, Unoccupied
        [4.0, 4.0, 4.0, 0.8],  # Batch 1, Occupied  T
    ], device=device)
    points2 = torch.tensor([
        [0.1, 0.1, 0.1, 0.7],  # Batch 0, Occupied  T
        [1.1, 1.1, 1.1, 0.3],  # Batch 0, Unoccupied
        [2.1, 2.1, 2.1, 0.7],  # Batch 0, Occupied  T
        [3.1, 3.1, 3.1, 0.8],  # Batch 1, Occupied  T
        [4.1, 4.1, 4.1, 0.3],  # Batch 1, Unoccupied
    ], device=device)
    batch_indices1 = torch.tensor([0, 0, 0, 1, 1, 1], device=device)
    batch_indices2 = torch.tensor([0, 0, 0, 1, 1], device=device)
    
    # Test with regular matching (not occupied-only)
    buffer = ChamferPointDataBuffer(occupancy_threshold=0.5)
    buffer.create_masks((points1, batch_indices1), (points2, batch_indices2))
    
    # Check occupied masks
    occupied_mask, occupied_mask_other = buffer.occupied_mask()
    expected_occupied_mask = torch.tensor([True, False, True, True, False, True], device=device)
    expected_occupied_mask_other = torch.tensor([True, False, True, True, False], device=device)
    assert torch.equal(occupied_mask, expected_occupied_mask)
    assert torch.equal(occupied_mask_other, expected_occupied_mask_other)
    
    mapping = buffer.mapping()
    expected_mapping = torch.tensor([[0, 2], [1, 1], [2, 0], [4, 3], [5, 4]], device=device)
    assert torch.equal(mapping, expected_mapping)

    mapped_mask, mapped_mask_other = buffer.mapped_mask()
    expected_mapped_mask = torch.tensor([True, True, True, False, True, True], device=device)
    expected_mapped_mask_other = torch.tensor([True, True, True, True, True], device=device)
    assert torch.equal(mapped_mask, expected_mapped_mask)
    assert torch.equal(mapped_mask_other, expected_mapped_mask_other)
    
    # Test with occupied-only matching
    buffer_occ_only = ChamferPointDataBuffer(occupancy_threshold=0.5, match_occupied_only=True)
    buffer_occ_only.create_masks((points1, batch_indices1), (points2, batch_indices2))

    occupied_mask, occupied_mask_other = buffer_occ_only.occupied_mask()
    expected_occupied_mask = torch.tensor([True, False, True, True, False, True], device=device)
    expected_occupied_mask_other = torch.tensor([True, False, True, True, False], device=device)
    assert torch.equal(occupied_mask, expected_occupied_mask)
    assert torch.equal(occupied_mask_other, expected_occupied_mask_other)

    # Check mapping - should only include occupied points
    mapping = buffer_occ_only.mapping()
    expected_mapping = torch.tensor([[0, 2], [2, 0], [5, 3]], device=device)
    assert torch.equal(mapping, expected_mapping)

    # Check mapped masks
    mapped_mask, mapped_mask_other = buffer_occ_only.mapped_mask()
    expected_mapped_mask = torch.tensor([True, False, True, False, False, True], device=device)
    expected_mapped_mask_other = torch.tensor([True, False, True, True, False], device=device)
    assert torch.equal(mapped_mask, expected_mapped_mask)
    assert torch.equal(mapped_mask_other, expected_mapped_mask_other)


# def test_sinkhorn_basic_matching(device):
#     # Create test data with clear matches
#     points1 = torch.tensor([
#         [0.0, 0.0, 0.0, 0.7],
#         [1.0, 1.0, 1.0, 0.3],
#         [2.0, 2.0, 2.0, 0.7],
#     ], device=device)
#     points2 = torch.tensor([
#         [0.1, 0.1, 0.1, 0.7],
#         [1.1, 1.1, 1.1, 0.3],
#         [2.1, 2.1, 2.1, 0.7],
#     ], device=device)
#     batch_indices1 = torch.tensor([0, 0, 0], device=device)
#     batch_indices2 = torch.tensor([0, 0, 0], device=device)
    
#     # Create buffer
#     buffer = SinkhornPointDataBuffer(occupancy_threshold=0.5)
#     buffer.create_masks((points1, batch_indices1), (points2, batch_indices2))
    
#     # Check masks
#     mapped_mask, mapped_mask_other = buffer.mapped_mask()
#     assert mapped_mask.shape == (3,)
#     assert mapped_mask_other.shape == (3,)
#     assert mapped_mask.sum().item() == mapped_mask_other.sum().item()
    
#     # Check mapping includes soft weights
#     mapping = buffer.mapping()
#     assert mapping.shape[1] == 3  # Includes soft weights
#     assert mapping.shape[0] == mapped_mask.sum().item()
    
#     # Check soft assignment properties
#     soft_assignment = buffer.soft_assignment()
#     assert soft_assignment.shape == (3, 3)
#     assert torch.allclose(soft_assignment.sum(dim=0), torch.ones(3, device=device))
#     assert torch.allclose(soft_assignment.sum(dim=1), torch.ones(3, device=device))


# def test_sinkhorn_temperature_effect(device):
#     # Test how temperature affects the soft assignment
#     points1 = torch.tensor([
#         [0.0, 0.0, 0.0, 0.7],
#         [1.0, 1.0, 1.0, 0.7],
#     ], device=device)
#     points2 = torch.tensor([
#         [0.1, 0.1, 0.1, 0.7],
#         [1.1, 1.1, 1.1, 0.7],
#     ], device=device)
#     batch_indices1 = torch.tensor([0, 0], device=device)
#     batch_indices2 = torch.tensor([0, 0], device=device)
    
#     # Test with low temperature (more discrete)
#     buffer_low_temp = SinkhornPointDataBuffer(temperature=0.01, occupancy_threshold=0.5)
#     buffer_low_temp.create_masks((points1, batch_indices1), (points2, batch_indices2))
#     soft_assignment_low = buffer_low_temp.soft_assignment()
    
#     # Test with high temperature (more uniform)
#     buffer_high_temp = SinkhornPointDataBuffer(temperature=1.0, occupancy_threshold=0.5)
#     buffer_high_temp.create_masks((points1, batch_indices1), (points2, batch_indices2))
#     soft_assignment_high = buffer_high_temp.soft_assignment()
    
#     # Lower temperature should give more extreme values
#     assert torch.max(soft_assignment_low) > torch.max(soft_assignment_high)
#     assert torch.min(soft_assignment_low) < torch.min(soft_assignment_high)


# def test_sinkhorn_batch_aware_matching(device):
#     # Create test data with multiple batches
#     points1 = torch.tensor([
#         [0.0, 0.0, 0.0, 0.7],  # Batch 0
#         [1.0, 1.0, 1.0, 0.7],  # Batch 1
#     ], device=device)
#     points2 = torch.tensor([
#         [0.1, 0.1, 0.1, 0.7],  # Batch 0
#         [1.1, 1.1, 1.1, 0.7],  # Batch 1
#     ], device=device)
#     batch_indices1 = torch.tensor([0, 1], device=device)
#     batch_indices2 = torch.tensor([0, 1], device=device)
    
#     buffer = SinkhornPointDataBuffer(occupancy_threshold=0.5)
#     buffer.create_masks((points1, batch_indices1), (points2, batch_indices2))
    
#     # Check mapping respects batch indices
#     mapping = buffer.mapping()
#     for i in range(len(mapping)):
#         idx1, idx2 = mapping[i, :2].long()
#         assert batch_indices1[idx1].item() == batch_indices2[idx2].item()
    
#     # Check soft assignment respects batches (with appropriate tolerance)
#     soft_assignment = buffer.soft_assignment()
#     for i in range(len(points1)):
#         for j in range(len(points2)):
#             if batch_indices1[i] != batch_indices2[j]:
#                 # Values should be very close to zero, but not exactly zero due to numerical precision
#                 assert soft_assignment[i, j].item() < 1e-5


# def test_sinkhorn_convergence(device):
#     # Test that different numbers of iterations converge to similar results
#     points1 = torch.tensor([
#         [0.0, 0.0, 0.0, 0.7],
#         [1.0, 1.0, 1.0, 0.7],
#     ], device=device)
#     points2 = torch.tensor([
#         [0.1, 0.1, 0.1, 0.7],
#         [1.1, 1.1, 1.1, 0.7],
#     ], device=device)
#     batch_indices1 = torch.tensor([0, 0], device=device)
#     batch_indices2 = torch.tensor([0, 0], device=device)
    
#     # Test with different numbers of iterations
#     buffer_low_iter = SinkhornPointDataBuffer(n_iters=10, occupancy_threshold=0.5)
#     buffer_low_iter.create_masks((points1, batch_indices1), (points2, batch_indices2))
#     soft_assignment_low = buffer_low_iter.soft_assignment()
    
#     buffer_high_iter = SinkhornPointDataBuffer(n_iters=50, occupancy_threshold=0.5)
#     buffer_high_iter.create_masks((points1, batch_indices1), (points2, batch_indices2))
#     soft_assignment_high = buffer_high_iter.soft_assignment()
    
#     # Results should be similar
#     assert torch.allclose(soft_assignment_low, soft_assignment_high, rtol=1e-3) 
