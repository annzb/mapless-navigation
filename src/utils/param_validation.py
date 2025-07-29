from typing import Union

import numpy as np
import torch


def validate_positive_int(value, value_name=None):
    name_str = f'{value_name} must be an integer' if value_name else 'value must be an integer'
    if not isinstance(value, int):
        raise ValueError(name_str)
    if value <= 0:
        raise ValueError(name_str + ' and positive')
    return value


def validate_positive_number(value, value_name=None):
    if not isinstance(value, (int, float)):
        raise ValueError(f'{value_name} must be a number')
    if value <= 0:
        raise ValueError(f'{value_name} must be positive')
    return value


def any_to_tensor(value: Union[np.ndarray, torch.Tensor], dtype=None, device=None):
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    elif not isinstance(value, torch.Tensor):
        raise ValueError(f'Invalid value type: {type(value)}, expected np.ndarray or torch.Tensor')
    
    if dtype:
        value = value.to(dtype=dtype)
    if device:
        value = value.to(device=device)
    return value
