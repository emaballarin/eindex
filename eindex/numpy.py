from typing import Any

import numpy as np

from . import _core
from ._core import Aggregation

__all__ = ["argmax", "argmin", "argsort", "gather", "gather_scatter", "scatter"]

Array = np.ndarray


class _NumpyIXP(_core.IXP):
    xp: Any

    def __init__(self) -> None:
        self.xp = np

    def permute_dims(self, arr, permutation):
        return np.transpose(arr, permutation)

    def arange_at_position(self, n_axes, axis, axis_len, array_to_copy_device_from):
        # array_to_copy_device_from is ignored as numpy supports only CPU
        x = np.arange(axis_len, dtype=np.int64)
        shape = [1] * n_axes
        shape[axis] = axis_len
        return np.reshape(x, shape)

    def scatter_aggregate(self, shape, flat_idx_1d, src_2d, agg, dtype):
        if agg == "sum":
            result = np.zeros(shape, dtype=dtype)
            np.add.at(result, flat_idx_1d, src_2d)
            return result
        if agg == "max":
            result = np.full(shape, fill_value=-np.inf, dtype=dtype)
            np.maximum.at(result, flat_idx_1d, src_2d)
            return result
        if agg == "min":
            result = np.full(shape, fill_value=np.inf, dtype=dtype)
            np.minimum.at(result, flat_idx_1d, src_2d)
            return result
        if agg == "mean":
            assert dtype in [np.float16, np.float32, np.float64], "mean reduction supported only for float tensors"
            nom = np.zeros(shape, dtype=dtype)
            np.add.at(nom, flat_idx_1d, src_2d)
            denom = np.zeros(shape, dtype=dtype)
            np.add.at(denom, flat_idx_1d, 1)
            return nom / denom
        raise NotImplementedError(agg)


_numpy_ixp = _NumpyIXP()


def argmax(tensor: Array, pattern: str, /) -> Array:
    formula = _core.ArgmaxFormula(pattern)
    return formula.apply_to_ixp(_numpy_ixp, tensor)


def argmin(tensor: Array, pattern: str, /) -> Array:
    formula = _core.ArgminFormula(pattern)
    return formula.apply_to_ixp(_numpy_ixp, tensor)


def argsort(tensor: Array, pattern: str, /, *, order_axis="order") -> Array:
    formula = _core.ArgsortFormula(pattern, order_axis=order_axis)
    return formula.apply_to_ixp(_numpy_ixp, tensor)


def _einindex(arr: Array, ind: Array | list[Array], pattern: str, /):
    formula = _core.IndexFormula(pattern)
    return formula.apply_to_numpy(_numpy_ixp, arr, ind)


def gather(arr: Array, ind: Array | list[Array], pattern: str, /, agg: Aggregation | None = None):
    formula = _core.GatherFormula(pattern=pattern, agg=agg)
    return formula.apply_to_numpy(_numpy_ixp, arr, ind)


def gather_scatter(arr: Array, ind: Array | list[Array], pattern: str, /, agg: Aggregation = "sum", **axis_sizes: int):
    formula = _core.GatherScatterFormula(pattern, agg=agg)
    return formula.apply_to_ixp(_numpy_ixp, arr, ind, axis_sizes=axis_sizes)


def scatter(arr: Array, ind: Array | list[Array], pattern: str, /, agg: Aggregation = "sum", **axis_sizes: int):
    formula = _core.ScatterFormula(pattern, agg=agg)
    return formula.apply_to_ixp(_numpy_ixp, arr, ind, axis_sizes=axis_sizes)
