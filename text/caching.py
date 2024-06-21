# Copyright © <2024> Eloï Alonso, Vincent Micheli
# SPDX-FileContributor: Eloï Alonso
# SPDX-FileContributor: Vincent Micheli
# SPDX-FileCopyright: Arnaud Pannatier
# SPDX-License-Identifier: GPL-3.0-only
# FileOrigin: https://github.com/eloialonso/iris/blob/main/src/models/kv_caching.py
"""KV-Caching from IRIS for minGPT."""

from typing import Tuple

import numpy as np
import torch


class Cache:
    """Cache for keys and values in attention layers."""

    def __init__(
        self,
        num_samples: int,
        num_heads: int,
        max_tokens: int,
        embed_dim: int,
        device: torch.device,
    ) -> None:
        """Initialize the cache."""
        assert embed_dim % num_heads == 0
        self._n, self._cache, self._size = num_samples, None, None
        self._reset = lambda n: torch.empty(
            n, num_heads, max_tokens, embed_dim // num_heads, device=device
        )  # (B, nh, T, hs)
        self.reset()

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        """Return the shape of the cache."""
        n, num_heads, _, head_dim = self._cache.shape
        return n, num_heads, self._size, head_dim

    def reset(self) -> None:
        """Reset the cache to empty value."""
        self._cache = self._reset(self._n)
        self._size = 0

    def prune(self, mask: np.ndarray) -> None:
        """Prune the cache with a mask."""
        assert mask.ndim == 1 and mask.shape[0] == self.shape[0]
        self._cache = self._cache[mask]
        self._n = self._cache.shape[0]

    def get(self) -> torch.Tensor:
        """Return the cache slicing at the right size."""
        return self._cache[:, :, : self._size, :]

    def update(self, x: torch.Tensor) -> None:
        """Update the cache with the new value, to be used before QKV operation."""
        assert (x.ndim == self._cache.ndim) and all(
            x.size(i) == self._cache.size(i) for i in (0, 1, 3)
        )
        assert self._size + x.size(2) <= self._cache.shape[2]
        self._cache = AssignWithoutInplaceCheck.apply(
            self._cache, x, 2, self._size, self._size + x.size(2)
        )
        self._size += x.size(2)


class KVCache:
    """Cache for keys and values in attention layers."""

    def __init__(
        self,
        n: int,
        num_heads: int,
        max_tokens: int,
        embed_dim: int,
        device: torch.device,
    ) -> None:
        """Initialize the cache. A Cache for keys and one for values."""
        self._k_cache = Cache(n, num_heads, max_tokens, embed_dim, device)
        self._v_cache = Cache(n, num_heads, max_tokens, embed_dim, device)

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        """Return the shape of the cache."""
        return self._k_cache.shape

    def reset(self) -> None:
        """Reset the caches to empty value."""
        self._k_cache.reset()
        self._v_cache.reset()

    def prune(self, mask: np.ndarray) -> None:
        """Prune the caches with a mask."""
        self._k_cache.prune(mask)
        self._v_cache.prune(mask)

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the caches slicing at the right size."""
        return self._k_cache.get(), self._v_cache.get()

    def update(self, k: torch.Tensor, v: torch.Tensor):
        """Update the caches with the new values, to be used before QKV operation."""
        self._k_cache.update(k)
        self._v_cache.update(v)


class KeysValues:
    """Holding the caches for the whole transformer."""

    def __init__(
        self,
        n: int,
        num_heads: int,
        max_tokens: int,
        embed_dim: int,
        num_layers: int,
        device: torch.device,
    ) -> None:
        """Initialize the caches for the whole transformer."""
        self._keys_values = tuple(
            [
                KVCache(n, num_heads, max_tokens, embed_dim, device)
                for _ in range(num_layers)
            ]
        )

    def __getitem__(self, key: int) -> KVCache:
        """Return the cache at the given layer."""
        return self._keys_values[key]

    def __len__(self):
        """Return the number of caches."""
        return len(self._keys_values)

    @property
    def size(self):
        """Return the size of the cache."""
        return self._keys_values[0].shape[2]

    def reset(self) -> None:
        """Reset all the caches to empty value."""
        for kv_cache in self._keys_values:
            kv_cache.reset()

    def prune(self, mask: np.ndarray) -> None:
        """Prune all the caches with a mask."""
        for kv_cache in self._keys_values:
            kv_cache.prune(mask)


class AssignWithoutInplaceCheck(torch.autograd.Function):
    """Update cache without additionnal check for autograd.

    Inspired from :
    https://discuss.pytorch.org/t/disable-in-place-correctness-version-check-any-other-workaround/90738/4
    Warning : do not use it to overwrite a slice twice.
    """

    @staticmethod
    def get_slice(dim: int, start: int, stop: int) -> Tuple[slice]:
        """Return the slice to update the cache."""
        return tuple(
            [
                slice(None),
            ]
            * dim
            + [slice(start, stop)]
        )

    @staticmethod
    def forward(
        ctx, input: torch.Tensor, value: torch.Tensor, dim: int, start: int, stop: int
    ) -> torch.Tensor:
        """Get the slice."""
        ctx.dim = dim
        ctx.start = start
        ctx.stop = stop
        input.data[AssignWithoutInplaceCheck.get_slice(dim, start, stop)] = value
        return input

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor]:
        """Return the gradient."""
        return (
            grad_out,
            grad_out[AssignWithoutInplaceCheck.get_slice(ctx.dim, ctx.start, ctx.stop)],
            None,
            None,
            None,
        )
