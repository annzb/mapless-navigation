import unittest
import torch
from metrics.loss_points import ChamferPointLoss
from metrics.data_buffer import ChamferPointDataBuffer


class TestChamferPointLoss(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 2
        
    def test_spatial_loss(self):
        # Create test data
        points1 = torch.tensor([
            [0.0, 0.0, 0.0, 0.7],  # Occupied
            [1.0, 1.0, 1.0, 0.3],  # Unoccupied
            [2.0, 2.0, 2.0, 0.7],  # Occupied
        ], device=self.device, requires_grad=True)
        points2 = torch.tensor([
            [0.1, 0.1, 0.1, 0.7],  # Occupied
            [1.1, 1.1, 1.1, 0.3],  # Unoccupied
            [2.1, 2.1, 2.1, 0.7],  # Occupied
        ], device=self.device, requires_grad=True)
        batch_indices1 = torch.tensor([0, 0, 0], device=self.device)
        batch_indices2 = torch.tensor([0, 0, 0], device=self.device)
        
        # Create buffer and loss
        buffer = ChamferPointDataBuffer(occupancy_threshold=0.5)
        buffer.create_masks((points1, batch_indices1), (points2, batch_indices2))
        loss_fn = ChamferPointLoss(batch_size=self.batch_size, device=self.device)
        loss_fn.to(self.device)  # Move loss function to correct device
        
        # Test spatial loss
        spatial_loss = loss_fn._calc_spatial_loss(
            (points1, batch_indices1),
            (points2, batch_indices2),
            buffer
        )
        
        # Check that loss is positive and requires gradients
        self.assertGreater(spatial_loss.item(), 0)
        self.assertTrue(spatial_loss.requires_grad)
        
        # Test gradient flow
        spatial_loss.backward()
        self.assertIsNotNone(points1.grad)
        self.assertIsNotNone(points2.grad)
        
    def test_probability_loss(self):
        # Create test data
        points1 = torch.tensor([
            [0.0, 0.0, 0.0, 0.7],  # Occupied
            [1.0, 1.0, 1.0, 0.3],  # Unoccupied
            [2.0, 2.0, 2.0, 0.7],  # Occupied
        ], device=self.device, requires_grad=True)
        points2 = torch.tensor([
            [0.1, 0.1, 0.1, 0.7],  # Occupied
            [1.1, 1.1, 1.1, 0.3],  # Unoccupied
            [2.1, 2.1, 2.1, 0.7],  # Occupied
        ], device=self.device, requires_grad=True)
        batch_indices1 = torch.tensor([0, 0, 0], device=self.device)
        batch_indices2 = torch.tensor([0, 0, 0], device=self.device)
        
        # Create buffer and loss
        buffer = ChamferPointDataBuffer(occupancy_threshold=0.5)
        buffer.create_masks((points1, batch_indices1), (points2, batch_indices2))
        loss_fn = ChamferPointLoss(batch_size=self.batch_size, device=self.device)
        loss_fn.to(self.device)  # Move loss function to correct device
        
        # Test probability loss
        prob_loss = loss_fn._calc_probability_loss(
            (points1, batch_indices1),
            (points2, batch_indices2),
            buffer
        )
        
        # Check that loss is positive and requires gradients
        self.assertGreater(prob_loss.item(), 0)
        self.assertTrue(prob_loss.requires_grad)
        
        # Test gradient flow
        prob_loss.backward()
        self.assertIsNotNone(points1.grad)
        self.assertIsNotNone(points2.grad)
        
    def test_empty_mapping(self):
        # Create test data with no matches
        points1 = torch.tensor([
            [0.0, 0.0, 0.0, 0.7],
            [1.0, 1.0, 1.0, 0.3],
        ], device=self.device, requires_grad=True)
        points2 = torch.tensor([
            [20.0, 20.0, 20.0, 0.7],  # Far away, no matches
            [21.0, 21.0, 21.0, 0.3],
        ], device=self.device, requires_grad=True)
        batch_indices1 = torch.tensor([0, 0], device=self.device)
        batch_indices2 = torch.tensor([0, 0], device=self.device)
        
        # Create buffer and loss
        buffer = ChamferPointDataBuffer(occupancy_threshold=0.5)
        buffer.create_masks((points1, batch_indices1), (points2, batch_indices2))
        loss_fn = ChamferPointLoss(batch_size=self.batch_size, device=self.device)
        loss_fn.to(self.device)  # Move loss function to correct device
        
        # Test both losses with empty mapping
        spatial_loss = loss_fn._calc_spatial_loss(
            (points1, batch_indices1),
            (points2, batch_indices2),
            buffer
        )
        prob_loss = loss_fn._calc_probability_loss(
            (points1, batch_indices1),
            (points2, batch_indices2),
            buffer
        )
        
        # Check that losses are max values and require gradients
        self.assertEqual(spatial_loss.item(), loss_fn.max_distance)
        self.assertEqual(prob_loss.item(), 1.0)
        self.assertTrue(spatial_loss.requires_grad)
        self.assertTrue(prob_loss.requires_grad)
        
        # Test gradient flow
        spatial_loss.backward()
        prob_loss.backward()
        self.assertIsNotNone(points1.grad)
        self.assertIsNotNone(points2.grad)
        
    def test_mixed_batches(self):
        # Create test data with some matched and some unmatched points
        points1 = torch.tensor([
            [0.0, 0.0, 0.0, 0.7],  # Matched
            [1.0, 1.0, 1.0, 0.3],  # Matched
            [10.0, 10.0, 10.0, 0.7],  # Unmatched
        ], device=self.device, requires_grad=True)
        points2 = torch.tensor([
            [0.1, 0.1, 0.1, 0.7],  # Matched
            [1.1, 1.1, 1.1, 0.3],  # Matched
            [20.0, 20.0, 20.0, 0.7],  # Unmatched
        ], device=self.device, requires_grad=True)
        batch_indices1 = torch.tensor([0, 0, 1], device=self.device)
        batch_indices2 = torch.tensor([0, 0, 1], device=self.device)
        
        # Create buffer and loss
        buffer = ChamferPointDataBuffer(occupancy_threshold=0.5)
        buffer.create_masks((points1, batch_indices1), (points2, batch_indices2))
        loss_fn = ChamferPointLoss(batch_size=self.batch_size, device=self.device)
        loss_fn.to(self.device)  # Move loss function to correct device
        
        # Test both losses
        spatial_loss = loss_fn._calc_spatial_loss(
            (points1, batch_indices1),
            (points2, batch_indices2),
            buffer
        )
        prob_loss = loss_fn._calc_probability_loss(
            (points1, batch_indices1),
            (points2, batch_indices2),
            buffer
        )
        
        # Check that losses are computed correctly
        self.assertGreater(spatial_loss.item(), 0)
        self.assertLess(spatial_loss.item(), loss_fn.max_distance)
        self.assertGreater(prob_loss.item(), 0)
        self.assertLess(prob_loss.item(), 1.0)
        self.assertTrue(spatial_loss.requires_grad)
        self.assertTrue(prob_loss.requires_grad)
        
        # Test gradient flow
        spatial_loss.backward()
        prob_loss.backward()
        self.assertIsNotNone(points1.grad)
        self.assertIsNotNone(points2.grad)


if __name__ == '__main__':
    unittest.main()
    