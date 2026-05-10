"""JAX backend for eindex.

Currently exposes ``argmin``, ``argmax``, ``argsort`` and ``gather``. The
backend works on ``jax.Array`` and is jit/grad/vmap/pmap safe — no custom
kernels.

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

__all__ = ["argmax", "argmin", "argsort", "gather"]


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
