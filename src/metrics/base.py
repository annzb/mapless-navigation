import torch
import torch.nn as nn

from metrics.data_buffer import OccupancyDataBuffer, PointOccupancyDataBuffer, MappedPointOccupancyDataBuffer


class BaseCriteria:
    def __init__(self, batch_size, device, **kwargs):
        super().__init__()
        if not isinstance(batch_size, int):
            raise ValueError('batch_size must be an integer')
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')
        if not isinstance(device, torch.device):
            raise ValueError('device must be a torch.device')
        
        self.default_value = 0.0
        self._batch_size = batch_size
        self._device = device

    def batch_size(self) -> int:
        return self._batch_size
    
    def device(self) -> torch.device:
        return self._device

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
        self.default_value = torch.tensor(float('inf'), device=self._device, requires_grad=True)


class BaseMetric(BaseCriteria):
    def __init__(self, name='', score_multiplier=1.0, negative=False, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(name, str):
            raise ValueError('Metric name must be a string')
        if not isinstance(score_multiplier, (int, float)):
            raise ValueError('Metric score_multiplier must be a number')
        if not isinstance(negative, bool):
            raise ValueError('Metric `negative` flag must be a boolean')
        
        self.total_score = 0.0
        self.best_score = 0.0
        self.name = (f'{name}_' if name else '') + 'metric_' + self.__class__.__name__.lower()
        self.negative = negative
        self.score_multiplier = score_multiplier
        self._scaled = False

    def reset_epoch(self):
        self.total_score = 0.0
        self._scaled = False

    def reset(self):
        self.reset_epoch()
        self.best_score = 0.0

    def scale_score(self, n_samples):
        if not isinstance(n_samples, int):
            raise ValueError('n_samples must be an integer')
        if n_samples <= 0:
            raise ValueError('n_samples must be positive')
        
        if self._scaled:
            raise RuntimeError(f'Metric {self.name} already scaled')
        self.total_score /= n_samples
        self._scaled = True
        save_condition = (self.total_score < self.best_score) if self.negative else (self.total_score > self.best_score)
        if save_condition:
            self.best_score = self.total_score

    def __call__(self, y_pred, y_true, *args, **kwargs):
        calc_result = self.forward(y_pred, y_true, *args, **kwargs) 
        if calc_result is None:
            return None
        
        score = calc_result * self.score_multiplier
        self.total_score += score
        if torch.is_tensor(score):
            return score.detach().item()
        return score


class OccupancyCriteria(BaseCriteria):
    def __init__(self, occupied_only, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(occupied_only, bool):
            raise ValueError('occupied_only must be a boolean')
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
    def __init__(self, max_point_distance, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(max_point_distance, (int, float)):
            raise ValueError('max_point_distance must be a number')
        if max_point_distance <= 0:
            raise ValueError('max_point_distance must be positive')
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
            if data_buffer._occupied_only:
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
            if data_buffer._occupied_only:
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
    
    def _calc_matching_ratios_soft(self, y_pred, y_true, data_buffer, target_masks):
        (y_pred_values, y_pred_batch_indices), (y_true_values, y_true_batch_indices) = y_pred, y_true
        mapped_mask_pred, mapped_mask_true = data_buffer.mapped_mask()
        pred_occ_mask, true_occ_mask = data_buffer.occupied_mask()
        
        ratios = []

        for b in range(self._batch_size):
            pred_mask, true_mask = target_masks[b]
            pred_vals, true_vals = y_pred_values[pred_mask], y_true_values[true_mask]

            if self.occupied_only:
                full_pred = y_pred_values[(y_pred_batch_indices == b) & pred_occ_mask]
                full_true = y_true_values[(y_true_batch_indices == b) & true_occ_mask]
            else:
                full_pred = y_pred_values[y_pred_batch_indices == b]
                full_true = y_true_values[y_true_batch_indices == b]

            total_conf_pred, total_conf_true = full_pred[:, 3].sum(), full_true[:, 3].sum()
            target_conf_pred, target_conf_true = pred_vals[:, 3].sum(), true_vals[:, 3].sum()

            ratio = (target_conf_pred + target_conf_true) / (total_conf_pred + total_conf_true + 1e-6)
            ratios.append(ratio)

        return torch.stack(ratios)

    # def _calc_matching_ratios_soft(self, y_pred, y_true, data_buffer, target_masks):
    #     (y_pred_values, y_pred_batch_indices), (y_true_values, y_true_batch_indices) = y_pred, y_true
    #     mapped_mask_pred, mapped_mask_true = data_buffer.mapped_mask()
    #     pred_occ_mask, true_occ_mask = data_buffer.occupied_mask()
        
    #     target_occupancy_pred, target_occupancy_true, total_occupancy_pred, total_occupancy_true = [], [], [], []

    #     for b in range(self._batch_size):
    #         pred_mask, true_mask = target_masks[b]
    #         pred_vals, true_vals = y_pred_values[pred_mask], y_true_values[true_mask]

    #         if self.occupied_only:
    #             full_pred = y_pred_values[(y_pred_batch_indices == b) & pred_occ_mask]
    #             full_true = y_true_values[(y_true_batch_indices == b) & true_occ_mask]
    #         else:
    #             full_pred = y_pred_values[y_pred_batch_indices == b]
    #             full_true = y_true_values[y_true_batch_indices == b]

    #         total_conf_pred, total_conf_true = full_pred[:, 3].sum(), full_true[:, 3].sum()
    #         target_conf_pred, target_conf_true = pred_vals[:, 3].sum(), true_vals[:, 3].sum()

    #         target_occupancy_pred.append(target_conf_pred)
    #         target_occupancy_true.append(target_conf_true)
    #         total_occupancy_pred.append(total_conf_pred)
    #         total_occupancy_true.append(total_conf_true)

    #     return torch.stack(target_occupancy_pred), torch.stack(target_occupancy_true), torch.stack(total_occupancy_pred), torch.stack(total_occupancy_true)


class GridOccupancyCriteria(OccupancyCriteria): pass
    # def filter_occupied(self, y_pred, y_true, *args, **kwargs):
    #     y_pred = y_pred[y_pred[:, :, -1] >= self.occupancy_threshold]
    #     y_true = y_true[y_true[:, :, -1] >= self.occupancy_threshold]
    #     return y_pred, y_true


class PointcloudOccupancyMetric(BaseMetric, PointcloudOccupancyCriteria): pass
class PointcloudOccupancyLoss(BaseLoss, PointcloudOccupancyCriteria): pass
class GridOccupancyMetric(BaseMetric, GridOccupancyCriteria): pass
class GridOccupancyLoss(BaseLoss, GridOccupancyCriteria): pass
