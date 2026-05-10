"""PyTorch backend for eindex.

Exposes ``argmin``, ``argmax``, ``argsort``, ``gather``, ``scatter`` and
``gather_scatter``. All ops are pure torch — no custom kernels — so the
user is free to wrap them in ``torch.compile`` or autograd as needed.

Autograd:
    - ``argmin`` / ``argmax`` / ``argsort`` produce integer tensors and have
      no useful gradient.
    - ``gather`` and ``scatter`` with ``agg='sum'`` / ``'mean'`` give exact
      gradients via ``index_add_``; ``agg='max'`` / ``'min'`` use the
      standard subgradient via ``scatter_reduce_``.

CUDA non-determinism:
    ``index_add_`` and ``scatter_reduce_`` are non-deterministic on CUDA
    (matches the torch ecosystem default). To force determinism, set
    ``torch.use_deterministic_algorithms(True)`` and accept the slowdown.
"""

from typing import Any
from typing import Protocol
from typing import TypeVar

import torch

from . import _core
from ._core import Aggregation

__all__ = ["argmax", "argmin", "argsort", "gather", "gather_scatter", "scatter"]


class ArrayProtocol(Protocol):
    @property
    def device(self) -> Any: ...


def np_take(input, indices, axis=None):
    if not isinstance(indices, torch.Tensor):
        indices = torch.tensor(indices, device=input.device)
    if axis is None:
        return input.flatten()[indices]
    if axis < 0:
        axis += input.ndim
    input = input.movedim(axis, 0)
    result = input[indices]
    if axis != 0:
        result = result.movedim(0, axis)
    return result


class _TorchXP:
    int64 = torch.int64
    float64 = torch.float64
    inf = torch.inf

    def reshape(self, x: torch.Tensor, shape: list[int]):
        return torch.reshape(x, shape)

    def full(self, shape: list[int], fill_value: float, dtype: torch.dtype | None = None):
        return torch.full(shape, fill_value, dtype=dtype)

    def zeros(self, shape: list[int], dtype: torch.dtype | None = None, device: torch.device | None = None):
        return torch.zeros(shape, dtype=dtype, device=device)

    def permute(self, x: torch.Tensor, permutation: list[int]):
        return torch.permute(x, permutation)

    def arange(self, axis_len: int, dtype: torch.dtype | None = None, device: torch.device | None = None):
        return torch.arange(axis_len, dtype=dtype, device=device)

    def all(self, x: torch.Tensor):
        return torch.all(torch.as_tensor(x))

    def broadcast_to(self, x: torch.Tensor, shape: list[int]):
        return torch.broadcast_to(x, shape)

    def take(self, x: torch.Tensor, indices: torch.Tensor, axis: int):
        return np_take(x, indices, axis=axis)

    def stack(self, arrays: list[torch.Tensor], axis=0):
        return torch.stack(arrays, dim=axis)

    def argmax(self, array, axis=None):
        return torch.argmax(array, dim=axis)

    def argmin(self, array, axis=None):
        return torch.argmin(array, dim=axis)

    def argsort(self, array, axis=None):
        if axis is None:
            return torch.argsort(array.flatten())
        return torch.argsort(array, dim=axis)

    def permute_dims(self, array: torch.Tensor, dims: list[int]):
        return array.permute(dims)

    def sort(self, array: torch.Tensor, axis: int = -1):
        return torch.sort(array, dim=axis).values

    def sum(self, array: torch.Tensor, axis=None):
        if axis is None:
            return array.sum()
        return array.sum(dim=axis)

    def mean(self, array: torch.Tensor, axis=None):
        if axis is None:
            return array.mean()
        return array.mean(dim=axis)

    def astype(self, array: torch.Tensor, dtype: torch.dtype):
        return array.to(dtype)

    def max(self, array: torch.Tensor, axis=None):
        if axis is None:
            # torch.max() with axis=None has a cryptic error about named dims.
            return array.max()
        if isinstance(axis, int):
            return array.max(dim=axis).values
        if isinstance(axis, (tuple, list)):
            for i in sorted(axis, reverse=True):
                array = array.max(dim=i).values
            return array
        raise TypeError(f"Unsupported axis type: {type(axis).__name__}")

    def min(self, array: torch.Tensor, axis=None):
        if axis is None:
            return array.min()
        if isinstance(axis, int):
            return array.min(dim=axis).values
        if isinstance(axis, (tuple, list)):
            for i in sorted(axis, reverse=True):
                array = array.min(dim=i).values
            return array
        raise TypeError(f"Unsupported axis type: {type(axis).__name__}")


class _TorchIXP(_core.IXP):
    def __init__(self) -> None:
        self.xp = _TorchXP()

    def permute_dims(self, arr, permutation):
        return self.xp.permute(arr, permutation)

    def arange_at_position(self, n_axes, axis, axis_len, array_to_copy_device_from: ArrayProtocol):
        xp = self.xp
        x = xp.arange(axis_len, dtype=xp.int64, device=array_to_copy_device_from.device)
        shape = [1] * n_axes
        shape[axis] = axis_len
        return xp.reshape(x, shape)

    def scatter_aggregate(self, shape, flat_idx_1d, src_2d, agg, dtype):
        device = src_2d.device
        # index_add_ / scatter_reduce_ require a long-typed index.
        flat_idx_1d = flat_idx_1d.to(torch.long)
        N, C = src_2d.shape
        if agg == "sum":
            result = torch.zeros(shape, dtype=dtype, device=device)
            result.index_add_(0, flat_idx_1d, src_2d)
            return result
        if agg == "mean":
            assert dtype.is_floating_point, "mean reduction supported only for float tensors"
            nom = torch.zeros(shape, dtype=dtype, device=device)
            nom.index_add_(0, flat_idx_1d, src_2d)
            denom = torch.zeros(shape, dtype=dtype, device=device)
            denom.index_add_(0, flat_idx_1d, torch.ones_like(src_2d))
            return nom / denom
        if agg in ("max", "min"):
            # ``include_self=False`` means the fill only matters for buckets that
            # are never written to. Float dtypes get ±inf; integer dtypes use the
            # representable extreme (numpy silently casts ±inf to the same value
            # via ``invalid value encountered in cast``).
            if dtype.is_floating_point:
                fill = float("-inf") if agg == "max" else float("inf")
            else:
                info = torch.iinfo(dtype)
                fill = info.min if agg == "max" else info.max
            result = torch.full(shape, fill, dtype=dtype, device=device)
            # scatter_reduce_'s autograd path needs index.shape == src.shape.
            idx_2d = flat_idx_1d.unsqueeze(-1).expand(N, C)
            reduce = "amax" if agg == "max" else "amin"
            result.scatter_reduce_(0, idx_2d, src_2d, reduce=reduce, include_self=False)
            return result
        raise NotImplementedError(agg)


Array = TypeVar("Array", bound=ArrayProtocol)


def argmax(tensor: Array, pattern: str, /) -> Array:  # noqa: UP047
    formula = _core.ArgmaxFormula(pattern)
    return formula.apply_to_ixp(_TorchIXP(), tensor)


def argmin(tensor: Array, pattern: str, /) -> Array:  # noqa: UP047
    formula = _core.ArgminFormula(pattern)
    return formula.apply_to_ixp(_TorchIXP(), tensor)


def argsort(tensor: Array, pattern: str, /, *, order_axis: str = "order") -> Array:  # noqa: UP047
    formula = _core.ArgsortFormula(pattern, order_axis=order_axis)
    return formula.apply_to_ixp(_TorchIXP(), tensor)


def _einindex(arr: Array, ind: Array | list[Array], pattern: str, /):  # noqa: UP047
    formula = _core.IndexFormula(pattern)
    return formula.apply_to_array_api(_TorchIXP(), arr, ind)


def gather(  # noqa: UP047
    arr: Array, ind: Array | list[Array], pattern: str, /, agg: Aggregation | None = None
):
    formula = _core.GatherFormula(pattern=pattern, agg=agg)
    return formula.apply_to_array_api(_TorchIXP(), arr, ind)


def scatter(  # noqa: UP047
    arr: Array, ind: Array | list[Array], pattern: str, /, agg: Aggregation = "sum", **axis_sizes: int
):
    formula = _core.ScatterFormula(pattern, agg=agg)
    return formula.apply_to_ixp(_TorchIXP(), arr, ind, axis_sizes=axis_sizes)


def gather_scatter(  # noqa: UP047
    arr: Array, ind: Array | list[Array], pattern: str, /, agg: Aggregation = "sum", **axis_sizes: int
):
    formula = _core.GatherScatterFormula(pattern, agg=agg)
    return formula.apply_to_ixp(_TorchIXP(), arr, ind, axis_sizes=axis_sizes)
