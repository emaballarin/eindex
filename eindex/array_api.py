"""Generic array-API backend for eindex.

Works on any array library that exposes an ``__array_namespace__()``
method conforming to the array-API standard. As of numpy 2.0, plain
``numpy`` qualifies; ``cupy``, ``array_api_strict`` and others do too.

Only ``argmin``, ``argmax``, ``argsort`` and ``gather`` are exposed —
``scatter`` and ``gather_scatter`` are not part of the array-API
standard, so they live in the per-library backends (numpy/torch/jax).
"""

from typing import Any
from typing import Protocol
from typing import TypeVar

from . import _core
from ._core import Aggregation

__all__ = ["argmax", "argmin", "argsort", "gather"]


class ArrayProtocol(Protocol):
    def __array_namespace__(self) -> Any: ...

    @property
    def device(self) -> Any: ...


class _ArrayApiIXP(_core.IXP):
    def __init__(self, xp) -> None:
        self.xp = xp

    def permute_dims(self, arr, permutation):
        return self.xp.permute_dims(arr, permutation)

    def arange_at_position(self, n_axes, axis, axis_len, array_to_copy_device_from: ArrayProtocol):
        xp = self.xp
        x = xp.arange(axis_len, dtype=xp.int64, device=array_to_copy_device_from.device)
        shape = [1] * n_axes
        shape[axis] = axis_len
        return xp.reshape(x, shape)


Array = TypeVar("Array", bound=ArrayProtocol)


def argmax(tensor: Array, pattern: str, /) -> Array:  # noqa: UP047
    formula = _core.ArgmaxFormula(pattern)
    ixp = _ArrayApiIXP(tensor.__array_namespace__())
    return formula.apply_to_ixp(ixp, tensor)


def argmin(tensor: Array, pattern: str, /) -> Array:  # noqa: UP047
    formula = _core.ArgminFormula(pattern)
    ixp = _ArrayApiIXP(tensor.__array_namespace__())
    return formula.apply_to_ixp(ixp, tensor)


def argsort(tensor: Array, pattern: str, /, *, order_axis: str = "order") -> Array:  # noqa: UP047
    formula = _core.ArgsortFormula(pattern, order_axis=order_axis)
    ixp = _ArrayApiIXP(tensor.__array_namespace__())
    return formula.apply_to_ixp(ixp, tensor)


def _einindex(arr: Array, ind: Array | list[Array], pattern: str, /):  # noqa: UP047
    formula = _core.IndexFormula(pattern)
    ixp = _ArrayApiIXP(arr.__array_namespace__())
    return formula.apply_to_array_api(ixp, arr, ind)


def gather(  # noqa: UP047
    arr: Array, ind: Array | list[Array], pattern: str, /, agg: Aggregation | None = None
):
    formula = _core.GatherFormula(pattern=pattern, agg=agg)
    ixp = _ArrayApiIXP(arr.__array_namespace__())
    return formula.apply_to_array_api(ixp, arr, ind)
