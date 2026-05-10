"""JAX backend for eindex.

Exposes ``argmin``, ``argmax``, ``argsort``, ``gather``, ``scatter`` and
``gather_scatter``. The backend works on ``jax.Array`` and is
jit/grad/vmap/pmap safe — no custom kernels.

Autograd:
    - ``argmin`` / ``argmax`` / ``argsort`` produce integer arrays and have
      no useful gradient.
    - ``gather`` / ``scatter`` with ``agg='sum'`` / ``'mean'`` give exact
      gradients via ``arr.at[idx].add(...)``; ``agg='max'`` / ``'min'`` use
      the standard subgradient via ``arr.at[idx].max(...)`` /
      ``arr.at[idx].min(...)``.

Caveat: when ``jax_enable_x64`` is False (the default), ``jnp.int64`` and
``jnp.float64`` silently become 32-bit. Indices are still correct as long
as the index space fits in ``int32``; for very large index spaces enable
the x64 flag (``jax.config.update("jax_enable_x64", True)``) before
calling these functions.
"""

from typing import Any
from typing import Protocol
from typing import TypeVar

import jax.numpy as jnp

from . import _core
from ._core import Aggregation

__all__ = ["argmax", "argmin", "argsort", "gather", "gather_scatter", "scatter"]


class ArrayProtocol(Protocol):
    @property
    def device(self) -> Any: ...


class _JaxXP:
    int64 = jnp.int64
    float64 = jnp.float64
    inf = jnp.inf

    def reshape(self, x, shape):
        return jnp.reshape(x, shape)

    def full(self, shape, fill_value, dtype=None):
        return jnp.full(shape, fill_value, dtype=dtype)

    def zeros(self, shape, dtype=None, device=None):
        # JAX places automatically; ``device`` is accepted for API symmetry only.
        del device
        return jnp.zeros(shape, dtype=dtype)

    def permute(self, x, permutation):
        return jnp.permute_dims(x, permutation)

    def permute_dims(self, x, permutation):
        return jnp.permute_dims(x, permutation)

    def arange(self, axis_len, dtype=None, device=None):
        # ``device`` is intentionally ignored: under jit, passing a concrete
        # device into a traced ``jnp.arange`` breaks tracing. JAX places the
        # array based on the surrounding computation.
        del device
        return jnp.arange(axis_len, dtype=dtype)

    def all(self, x):
        return jnp.all(x)

    def broadcast_to(self, x, shape):
        return jnp.broadcast_to(x, shape)

    def take(self, x, indices, axis):
        return jnp.take(x, indices, axis=axis)

    def stack(self, arrays, axis=0):
        return jnp.stack(arrays, axis=axis)

    def argmax(self, array, axis=None):
        return jnp.argmax(array, axis=axis)

    def argmin(self, array, axis=None):
        return jnp.argmin(array, axis=axis)

    def argsort(self, array, axis=None):
        if axis is None:
            return jnp.argsort(jnp.reshape(array, (-1,)))
        return jnp.argsort(array, axis=axis)

    def sort(self, array, axis=-1):
        return jnp.sort(array, axis=axis)

    def sum(self, array, axis=None):
        return jnp.sum(array, axis=axis)

    def mean(self, array, axis=None):
        return jnp.mean(array, axis=axis)

    def astype(self, array, dtype):
        return array.astype(dtype)

    def max(self, array, axis=None):
        # ``jnp.max`` natively accepts an int / tuple / None, no looping needed.
        return jnp.max(array, axis=axis)

    def min(self, array, axis=None):
        return jnp.min(array, axis=axis)


class _JaxIXP(_core.IXP):
    def __init__(self) -> None:
        self.xp = _JaxXP()

    def permute_dims(self, arr, permutation):
        return jnp.permute_dims(arr, permutation)

    def arange_at_position(self, n_axes, axis, axis_len, array_to_copy_device_from):
        del array_to_copy_device_from  # JAX places automatically
        x = jnp.arange(axis_len, dtype=jnp.int64)
        shape = [1] * n_axes
        shape[axis] = axis_len
        return jnp.reshape(x, shape)

    def scatter_aggregate(self, shape, flat_idx_1d, src_2d, agg, dtype):
        # ``mode='drop'`` silently ignores out-of-bounds indices. The pattern
        # parser already prevents OOB; this is just cheap insurance.
        if agg == "sum":
            return jnp.zeros(shape, dtype=dtype).at[flat_idx_1d].add(src_2d, mode="drop")
        if agg == "mean":
            assert jnp.issubdtype(dtype, jnp.floating), "mean reduction supported only for float tensors"
            nom = jnp.zeros(shape, dtype=dtype).at[flat_idx_1d].add(src_2d, mode="drop")
            denom = jnp.zeros(shape, dtype=dtype).at[flat_idx_1d].add(jnp.ones_like(src_2d), mode="drop")
            return nom / denom
        if agg in ("max", "min"):
            # ±inf is not representable in integer dtypes; use the dtype's
            # extreme so the cast is well-defined. Buckets that get scattered
            # to override this; unreached buckets keep the sentinel.
            if jnp.issubdtype(dtype, jnp.floating):
                fill = -jnp.inf if agg == "max" else jnp.inf
            else:
                info = jnp.iinfo(dtype)
                fill = info.min if agg == "max" else info.max
            init = jnp.full(shape, fill, dtype=dtype)
            if agg == "max":
                return init.at[flat_idx_1d].max(src_2d, mode="drop")
            return init.at[flat_idx_1d].min(src_2d, mode="drop")
        raise NotImplementedError(agg)


Array = TypeVar("Array", bound=ArrayProtocol)


def argmax(tensor: Array, pattern: str, /) -> Array:  # noqa: UP047
    formula = _core.ArgmaxFormula(pattern)
    return formula.apply_to_ixp(_JaxIXP(), tensor)


def argmin(tensor: Array, pattern: str, /) -> Array:  # noqa: UP047
    formula = _core.ArgminFormula(pattern)
    return formula.apply_to_ixp(_JaxIXP(), tensor)


def argsort(tensor: Array, pattern: str, /, *, order_axis: str = "order") -> Array:  # noqa: UP047
    formula = _core.ArgsortFormula(pattern, order_axis=order_axis)
    return formula.apply_to_ixp(_JaxIXP(), tensor)


def _einindex(arr: Array, ind: Array | list[Array], pattern: str, /):  # noqa: UP047
    formula = _core.IndexFormula(pattern)
    return formula.apply_to_array_api(_JaxIXP(), arr, ind)


def gather(  # noqa: UP047
    arr: Array, ind: Array | list[Array], pattern: str, /, agg: Aggregation | None = None
):
    formula = _core.GatherFormula(pattern=pattern, agg=agg)
    return formula.apply_to_array_api(_JaxIXP(), arr, ind)


def scatter(  # noqa: UP047
    arr: Array, ind: Array | list[Array], pattern: str, /, agg: Aggregation = "sum", **axis_sizes: int
):
    formula = _core.ScatterFormula(pattern, agg=agg)
    return formula.apply_to_ixp(_JaxIXP(), arr, ind, axis_sizes=axis_sizes)


def gather_scatter(  # noqa: UP047
    arr: Array, ind: Array | list[Array], pattern: str, /, agg: Aggregation = "sum", **axis_sizes: int
):
    formula = _core.GatherScatterFormula(pattern, agg=agg)
    return formula.apply_to_ixp(_JaxIXP(), arr, ind, axis_sizes=axis_sizes)
