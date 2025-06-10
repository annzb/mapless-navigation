from typing import Union


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
