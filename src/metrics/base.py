import torch
import torch.nn as nn

from metrics.data_buffer import OccupancyDataBuffer, PointOccupancyDataBuffer, MappedPointOccupancyDataBuffer


class BaseCriteria:
    def __init__(self, batch_size=0, device=None, **kwargs):
        super().__init__()
        self.default_value = 0.0
        self._batch_size = batch_size
        self.device = device

    def _validate_input(self, y_pred, y_true, *args, **kwargs):
        return len(y_pred) != 0 or len(y_true) != 0, 'Empty inputs.'

    def _calc(self, y_pred, y_true, *args, **kwargs):
        return self.default_value

    def forward(self, y_pred, y_true, *args, **kwargs):
        valid, error = self._validate_input(y_pred, y_true, *args, **kwargs)
        if valid:
            return self._calc(y_pred, y_true, *args, **kwargs)
        raise RuntimeError(error)


class BaseLoss(BaseCriteria, nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.default_value = torch.tensor(float('inf'), device=self.device, requires_grad=True)


class BaseMetric(BaseCriteria):
    def __init__(self, name='', score_multiplier=1.0, **kwargs):
        super().__init__(**kwargs)
        self.total_score = 0.0
        self.best_score = 0.0
        self.name = (f'{name}_' if name else '') + self.__class__.__name__.lower()
        self._scaled = False
        self.score_multiplier = score_multiplier

    def reset_epoch(self):
        self.total_score = 0.0
        self._scaled = False

    def reset(self):
        self.reset_epoch()
        self.best_score = 0.0

    def scale_score(self, n_samples):
        if self._scaled:
            raise RuntimeError(f'Metric {self.name} already scaled')
        self.total_score /= n_samples
        self._scaled = True
        if self.total_score > self.best_score:
            self.best_score = self.total_score

    def __call__(self, y_pred, y_true, *args, **kwargs):
        score = self.forward(y_pred, y_true, *args, **kwargs) * self.score_multiplier
        self.total_score += score
        return score.detach().item()


class OccupancyCriteria(BaseCriteria):
    def __init__(self, occupied_only=False, **kwargs):
        super().__init__(**kwargs)
        self.occupied_only = occupied_only

    def _validate_input(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        valid, error = super()._validate_input(y_pred, y_true, data_buffer=data_buffer, *args, **kwargs)
        if valid:
            if data_buffer is None or not isinstance(data_buffer, OccupancyDataBuffer):
                valid, error = False, f'Data buffer not available. Expected instance of {OccupancyDataBuffer.__name__}, got {type(data_buffer).__name__}'
        if valid:
            if data_buffer.occupied_mask() is None:
                valid, error = False, 'Occupancy data not available in data buffer.'
        return valid, error


class PointcloudOccupancyCriteria(OccupancyCriteria):
    def __init__(self, max_point_distance=10, **kwargs):
        super().__init__(**kwargs)
        self.max_distance = max_point_distance

    def _validate_input(self, y_pred, y_true, data_buffer=None, *args, **kwargs):
        valid, error = super()._validate_input(y_pred, y_true, data_buffer=data_buffer, *args, **kwargs)
        if valid:
            if data_buffer is None or not isinstance(data_buffer, (PointOccupancyDataBuffer, MappedPointOccupancyDataBuffer)):
                valid, error = False, f'Data buffer not available. Expected instance of {PointOccupancyDataBuffer.__name__} or {MappedPointOccupancyDataBuffer.__name__}, got {type(data_buffer).__name__}'
        if valid:
            if isinstance(data_buffer, MappedPointOccupancyDataBuffer):
                if data_buffer.mapped_mask() is None:
                    valid, error = False, 'Mapped masks not available in data buffer.'
                if data_buffer.mapping() is None:
                    valid, error = False, 'Point mapping not available in data buffer.'
        return valid, error
    
    def _calc_matched_masks(self, y_pred_batch_indices, y_true_batch_indices, data_buffer):
        mapped_mask, mapped_mask_other = data_buffer.mapped_mask()
        pred_occ_mask, true_occ_mask = data_buffer.occupied_mask()
        masks = []
        
        for b in range(self._batch_size):
            pred_mask = (y_pred_batch_indices == b) & mapped_mask
            true_mask = (y_true_batch_indices == b) & mapped_mask_other
            if data_buffer._match_occupied_only:
                pred_mask &= pred_occ_mask
                true_mask &= true_occ_mask
            masks.append((pred_mask, true_mask))

        return masks
    
    def _calc_unmatched_masks(self, y_pred_batch_indices, y_true_batch_indices, data_buffer):
        mapped_mask, mapped_mask_other = data_buffer.mapped_mask()
        pred_occ_mask, true_occ_mask = data_buffer.occupied_mask()
        masks = []
        
        for b in range(self._batch_size):
            pred_mask = (y_pred_batch_indices == b) & ~mapped_mask
            true_mask = (y_true_batch_indices == b) & ~mapped_mask_other
            if data_buffer._match_occupied_only:
                pred_mask &= pred_occ_mask
                true_mask &= true_occ_mask
            masks.append((pred_mask, true_mask))

        return masks
    
    def _calc_matching_ratios(self, y_pred_batch_indices, y_true_batch_indices, data_buffer, target_masks):
        pred_occ_mask, true_occ_mask = data_buffer.occupied_mask()
        ratios = []
        
        for b in range(self._batch_size):
            pred_mask, true_mask = target_masks[b]
            n_target_pred, n_target_true = pred_mask.sum(), true_mask.sum()

            if self.occupied_only:
                n_pred, n_true = ((y_pred_batch_indices == b) & pred_occ_mask).sum(), ((y_true_batch_indices == b) & true_occ_mask).sum()
            else:
                n_pred, n_true = (y_pred_batch_indices == b).sum(), (y_true_batch_indices == b).sum()

            total_points = n_pred + n_true + 1e-6
            ratio = (n_target_pred + n_target_true) / total_points
            ratios.append(ratio)

        return torch.stack(ratios)


class GridOccupancyCriteria(OccupancyCriteria): pass
    # def filter_occupied(self, y_pred, y_true, *args, **kwargs):
    #     y_pred = y_pred[y_pred[:, :, -1] >= self.occupancy_threshold]
    #     y_true = y_true[y_true[:, :, -1] >= self.occupancy_threshold]
    #     return y_pred, y_true


class PointcloudOccupancyMetric(BaseMetric, PointcloudOccupancyCriteria): pass
class PointcloudOccupancyLoss(BaseLoss, PointcloudOccupancyCriteria): pass
class GridOccupancyMetric(BaseMetric, GridOccupancyCriteria): pass
class GridOccupancyLoss(BaseLoss, GridOccupancyCriteria): pass
