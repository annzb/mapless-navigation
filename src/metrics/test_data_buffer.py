import unittest
import torch
from metrics.data_buffer import (
    OccupancyDataBuffer,
    PointOccupancyDataBuffer,
    MappedPointOccupancyDataBuffer,
    ChamferPointDataBuffer,
    SinkhornPointDataBuffer
)


class TestPointOccupancyBuffer(unittest.TestCase):
    def test_point_occupancy_buffer(self):
        # Create test data
        points = torch.tensor([
            [0.0, 0.0, 0.0, 0.3],  # Below threshold
            [1.0, 1.0, 1.0, 0.7],  # Above threshold
            [2.0, 2.0, 2.0, 0.5],  # At threshold
        ])
        batch_indices = torch.tensor([0, 0, 0])
        
        # Create buffer
        buffer = PointOccupancyDataBuffer(occupancy_threshold=0.5)
        buffer.create_masks((points, batch_indices))
        
        # Check occupied mask
        occupied_mask = buffer.occupied_mask()
        self.assertEqual(occupied_mask.shape, (3,))
        self.assertEqual(occupied_mask[0].item(), False)
        self.assertEqual(occupied_mask[1].item(), True)
        self.assertEqual(occupied_mask[2].item(), True)


class TestChamferBuffer(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_basic_matching(self):
        # Create test data with clear matches
        points1 = torch.tensor([
            [0.0, 0.0, 0.0, 0.7],
            [1.0, 1.0, 1.0, 0.3],
            [2.0, 2.0, 2.0, 0.7],
        ], device=self.device)
        points2 = torch.tensor([
            [0.1, 0.1, 0.1, 0.7],
            [1.1, 1.1, 1.1, 0.3],
            [2.1, 2.1, 2.1, 0.7],
        ], device=self.device)
        batch_indices1 = torch.tensor([0, 0, 0], device=self.device)
        batch_indices2 = torch.tensor([0, 0, 0], device=self.device)
        
        # Create buffer
        buffer = ChamferPointDataBuffer(occupancy_threshold=0.5)
        buffer.create_masks((points1, batch_indices1), (points2, batch_indices2))
        
        # Check masks
        mapped_mask, mapped_mask_other = buffer.mapped_mask()
        self.assertEqual(mapped_mask.shape, (3,))
        self.assertEqual(mapped_mask_other.shape, (3,))
        self.assertEqual(mapped_mask.sum().item(), mapped_mask_other.sum().item())
        
        # Check mapping
        mapping = buffer.mapping()
        self.assertEqual(mapping.shape[1], 2)
        self.assertEqual(mapping.shape[0], mapped_mask.sum().item())

    def test_empty_mapping(self):
        # Create test data with no possible matches (points too far apart)
        points1 = torch.tensor([
            [0.0, 0.0, 0.0, 0.7],
            [1.0, 1.0, 1.0, 0.3],
        ], device=self.device)
        points2 = torch.tensor([
            [10.0, 10.0, 10.0, 0.7],
            [11.0, 11.0, 11.0, 0.3],
        ], device=self.device)
        batch_indices1 = torch.tensor([0, 0], device=self.device)
        batch_indices2 = torch.tensor([0, 0], device=self.device)
        
        buffer = ChamferPointDataBuffer(occupancy_threshold=0.5)
        buffer.create_masks((points1, batch_indices1), (points2, batch_indices2))
        
        # Check empty mapping
        mapping = buffer.mapping()
        self.assertEqual(mapping.numel(), 0)
        
        # Check empty masks
        mapped_mask, mapped_mask_other = buffer.mapped_mask()
        self.assertEqual(mapped_mask.sum().item(), 0)
        self.assertEqual(mapped_mask_other.sum().item(), 0)
        
        # Check empty occupied mapped masks
        occ_mapped_mask, occ_mapped_mask_other = buffer.occupied_mapped_mask()
        self.assertEqual(occ_mapped_mask.sum().item(), 0)
        self.assertEqual(occ_mapped_mask_other.sum().item(), 0)

    def test_batch_aware_matching(self):
        # Create test data with multiple batches
        points1 = torch.tensor([
            [0.0, 0.0, 0.0, 0.7],  # Batch 0
            [1.0, 1.0, 1.0, 0.7],  # Batch 1
        ], device=self.device)
        points2 = torch.tensor([
            [0.1, 0.1, 0.1, 0.7],  # Batch 0
            [1.1, 1.1, 1.1, 0.7],  # Batch 1
        ], device=self.device)
        batch_indices1 = torch.tensor([0, 1], device=self.device)
        batch_indices2 = torch.tensor([0, 1], device=self.device)
        
        buffer = ChamferPointDataBuffer(occupancy_threshold=0.5)
        buffer.create_masks((points1, batch_indices1), (points2, batch_indices2))
        
        # Check mapping respects batch indices
        mapping = buffer.mapping()
        for i in range(len(mapping)):
            idx1, idx2 = mapping[i, :2]
            self.assertEqual(batch_indices1[idx1].item(), batch_indices2[idx2].item())
        
        # Check masks are consistent within batches
        mapped_mask, mapped_mask_other = buffer.mapped_mask()
        self.assertEqual(mapped_mask.sum().item(), mapped_mask_other.sum().item())
        
        # Check occupied mapped masks are consistent
        occ_mapped_mask, occ_mapped_mask_other = buffer.occupied_mapped_mask()
        self.assertEqual(occ_mapped_mask.sum().item(), occ_mapped_mask_other.sum().item())

    def test_single_point_matching(self):
        # Test with single point in each cloud
        points1 = torch.tensor([[0.0, 0.0, 0.0, 0.7]], device=self.device)
        points2 = torch.tensor([[0.1, 0.1, 0.1, 0.7]], device=self.device)
        batch_indices1 = torch.tensor([0], device=self.device)
        batch_indices2 = torch.tensor([0], device=self.device)
        
        buffer = ChamferPointDataBuffer(occupancy_threshold=0.5)
        buffer.create_masks((points1, batch_indices1), (points2, batch_indices2))
        
        # Check single point mapping
        mapping = buffer.mapping()
        self.assertEqual(mapping.shape[0], 1)
        self.assertEqual(mapping.shape[1], 2)
        
        # Check masks
        mapped_mask, mapped_mask_other = buffer.mapped_mask()
        self.assertEqual(mapped_mask.sum().item(), 1)
        self.assertEqual(mapped_mask_other.sum().item(), 1)

    def test_mixed_occupancy_matching(self):
        # Test with mixed occupancy values
        points1 = torch.tensor([
            [0.0, 0.0, 0.0, 0.7],  # Occupied
            [1.0, 1.0, 1.0, 0.3],  # Unoccupied
            [2.0, 2.0, 2.0, 0.7],  # Occupied
        ], device=self.device)
        points2 = torch.tensor([
            [0.1, 0.1, 0.1, 0.7],  # Occupied
            [1.1, 1.1, 1.1, 0.3],  # Unoccupied
            [2.1, 2.1, 2.1, 0.7],  # Occupied
        ], device=self.device)
        batch_indices1 = torch.tensor([0, 0, 0], device=self.device)
        batch_indices2 = torch.tensor([0, 0, 0], device=self.device)
        
        # Test with regular matching (not occupied-only)
        buffer = ChamferPointDataBuffer(occupancy_threshold=0.5)
        buffer.create_masks((points1, batch_indices1), (points2, batch_indices2))
        
        # Check that both occupied and unoccupied points can be matched
        mapped_mask, mapped_mask_other = buffer.mapped_mask()
        occ_mapped_mask, occ_mapped_mask_other = buffer.occupied_mapped_mask()
        
        # Occupied mapped masks should be a subset of mapped masks
        self.assertTrue(torch.all(occ_mapped_mask <= mapped_mask))
        self.assertTrue(torch.all(occ_mapped_mask_other <= mapped_mask_other))
        
        # Test with occupied-only matching
        buffer_occ_only = ChamferPointDataBuffer(occupancy_threshold=0.5, match_occupied_only=True)
        buffer_occ_only.create_masks((points1, batch_indices1), (points2, batch_indices2))
        
        # Check that only occupied points are matched
        mapped_mask, mapped_mask_other = buffer_occ_only.mapped_mask()
        occ_mask1, occ_mask2 = buffer_occ_only.occupied_mask()
        
        self.assertTrue(torch.equal(mapped_mask & ~occ_mask1, torch.zeros_like(mapped_mask)))
        self.assertTrue(torch.equal(mapped_mask_other & ~occ_mask2, torch.zeros_like(mapped_mask_other)))


class TestSinkhornBuffer(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_basic_matching(self):
        # Create test data with clear matches
        points1 = torch.tensor([
            [0.0, 0.0, 0.0, 0.7],
            [1.0, 1.0, 1.0, 0.3],
            [2.0, 2.0, 2.0, 0.7],
        ], device=self.device)
        points2 = torch.tensor([
            [0.1, 0.1, 0.1, 0.7],
            [1.1, 1.1, 1.1, 0.3],
            [2.1, 2.1, 2.1, 0.7],
        ], device=self.device)
        batch_indices1 = torch.tensor([0, 0, 0], device=self.device)
        batch_indices2 = torch.tensor([0, 0, 0], device=self.device)
        
        # Create buffer
        buffer = SinkhornPointDataBuffer(occupancy_threshold=0.5)
        buffer.create_masks((points1, batch_indices1), (points2, batch_indices2))
        
        # Check masks
        mapped_mask, mapped_mask_other = buffer.mapped_mask()
        self.assertEqual(mapped_mask.shape, (3,))
        self.assertEqual(mapped_mask_other.shape, (3,))
        self.assertEqual(mapped_mask.sum().item(), mapped_mask_other.sum().item())
        
        # Check mapping includes soft weights
        mapping = buffer.mapping()
        self.assertEqual(mapping.shape[1], 3)  # Includes soft weights
        self.assertEqual(mapping.shape[0], mapped_mask.sum().item())
        
        # Check soft assignment properties
        soft_assignment = buffer.soft_assignment()
        self.assertEqual(soft_assignment.shape, (3, 3))
        self.assertTrue(torch.allclose(soft_assignment.sum(dim=0), torch.ones(3, device=self.device)))
        self.assertTrue(torch.allclose(soft_assignment.sum(dim=1), torch.ones(3, device=self.device)))

    def test_temperature_effect(self):
        # Test how temperature affects the soft assignment
        points1 = torch.tensor([
            [0.0, 0.0, 0.0, 0.7],
            [1.0, 1.0, 1.0, 0.7],
        ], device=self.device)
        points2 = torch.tensor([
            [0.1, 0.1, 0.1, 0.7],
            [1.1, 1.1, 1.1, 0.7],
        ], device=self.device)
        batch_indices1 = torch.tensor([0, 0], device=self.device)
        batch_indices2 = torch.tensor([0, 0], device=self.device)
        
        # Test with low temperature (more discrete)
        buffer_low_temp = SinkhornPointDataBuffer(temperature=0.01, occupancy_threshold=0.5)
        buffer_low_temp.create_masks((points1, batch_indices1), (points2, batch_indices2))
        soft_assignment_low = buffer_low_temp.soft_assignment()
        
        # Test with high temperature (more uniform)
        buffer_high_temp = SinkhornPointDataBuffer(temperature=1.0, occupancy_threshold=0.5)
        buffer_high_temp.create_masks((points1, batch_indices1), (points2, batch_indices2))
        soft_assignment_high = buffer_high_temp.soft_assignment()
        
        # Lower temperature should give more extreme values
        self.assertTrue(torch.max(soft_assignment_low) > torch.max(soft_assignment_high))
        self.assertTrue(torch.min(soft_assignment_low) < torch.min(soft_assignment_high))

    def test_batch_aware_matching(self):
        # Create test data with multiple batches
        points1 = torch.tensor([
            [0.0, 0.0, 0.0, 0.7],  # Batch 0
            [1.0, 1.0, 1.0, 0.7],  # Batch 1
        ], device=self.device)
        points2 = torch.tensor([
            [0.1, 0.1, 0.1, 0.7],  # Batch 0
            [1.1, 1.1, 1.1, 0.7],  # Batch 1
        ], device=self.device)
        batch_indices1 = torch.tensor([0, 1], device=self.device)
        batch_indices2 = torch.tensor([0, 1], device=self.device)
        
        buffer = SinkhornPointDataBuffer(occupancy_threshold=0.5)
        buffer.create_masks((points1, batch_indices1), (points2, batch_indices2))
        
        # Check mapping respects batch indices
        mapping = buffer.mapping()
        for i in range(len(mapping)):
            idx1, idx2 = mapping[i, :2].long()
            self.assertEqual(batch_indices1[idx1].item(), batch_indices2[idx2].item())
        
        # Check soft assignment respects batches (with appropriate tolerance)
        soft_assignment = buffer.soft_assignment()
        for i in range(len(points1)):
            for j in range(len(points2)):
                if batch_indices1[i] != batch_indices2[j]:
                    # Values should be very close to zero, but not exactly zero due to numerical precision
                    self.assertLess(soft_assignment[i, j].item(), 1e-5)

    def test_convergence(self):
        # Test that different numbers of iterations converge to similar results
        points1 = torch.tensor([
            [0.0, 0.0, 0.0, 0.7],
            [1.0, 1.0, 1.0, 0.7],
        ], device=self.device)
        points2 = torch.tensor([
            [0.1, 0.1, 0.1, 0.7],
            [1.1, 1.1, 1.1, 0.7],
        ], device=self.device)
        batch_indices1 = torch.tensor([0, 0], device=self.device)
        batch_indices2 = torch.tensor([0, 0], device=self.device)
        
        # Test with different numbers of iterations
        buffer_low_iter = SinkhornPointDataBuffer(n_iters=10, occupancy_threshold=0.5)
        buffer_low_iter.create_masks((points1, batch_indices1), (points2, batch_indices2))
        soft_assignment_low = buffer_low_iter.soft_assignment()
        
        buffer_high_iter = SinkhornPointDataBuffer(n_iters=50, occupancy_threshold=0.5)
        buffer_high_iter.create_masks((points1, batch_indices1), (points2, batch_indices2))
        soft_assignment_high = buffer_high_iter.soft_assignment()
        
        # Results should be similar
        self.assertTrue(torch.allclose(soft_assignment_low, soft_assignment_high, rtol=1e-3))


class TestOccupiedOnlyMatching(unittest.TestCase):
    def test_occupied_only_matching(self):
        # Create test data with some unoccupied points
        points1 = torch.tensor([
            [0.0, 0.0, 0.0, 0.7],  # Occupied
            [1.0, 1.0, 1.0, 0.3],  # Unoccupied
            [2.0, 2.0, 2.0, 0.7],  # Occupied
        ])
        points2 = torch.tensor([
            [0.1, 0.1, 0.1, 0.7],  # Occupied
            [1.1, 1.1, 1.1, 0.3],  # Unoccupied
            [2.1, 2.1, 2.1, 0.7],  # Occupied
        ])
        batch_indices1 = torch.tensor([0, 0, 0])
        batch_indices2 = torch.tensor([0, 0, 0])
        
        # Create buffer with occupied-only matching
        buffer = ChamferPointDataBuffer(occupancy_threshold=0.5, match_occupied_only=True)
        buffer.create_masks((points1, batch_indices1), (points2, batch_indices2))
        
        # Check masks
        mapped_mask, mapped_mask_other = buffer.mapped_mask()
        occupied_mapped_mask, occupied_mapped_mask_other = buffer.occupied_mapped_mask()
        
        # For occupied-only matching, mapped masks should be the same as occupied mapped masks
        self.assertTrue(torch.allclose(mapped_mask, occupied_mapped_mask))
        self.assertTrue(torch.allclose(mapped_mask_other, occupied_mapped_mask_other))
        
        # Check that only occupied points are matched
        self.assertEqual(mapped_mask[1].item(), False)  # Unoccupied point should not be matched
        self.assertEqual(mapped_mask_other[1].item(), False)  # Unoccupied point should not be matched


if __name__ == '__main__':
    unittest.main() 
