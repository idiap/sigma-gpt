# Copyright © <2024> Idiap Research Institute <contact@idiap.ch>

# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>

# SPDX-License-Identifier: AGPL-3.0-only
"""Ordering functions for σ-GPT models."""

from collections import deque
from functools import cache

import torch


def reorder(x, order, reverse=False):  # x is NxTxD1x...xDk, order is NxT'
    """Reorder the sequence.

    Args:
        x (torch.Tensor): Input tensor.
        order (torch.Tensor): Order tensor.
        reverse (bool): Whether to reverse the order.

    Returns:
        torch.Tensor: Reordered tensor.
    """
    u = x.reshape(x.size()[:2] + (-1,))
    order = order.unsqueeze(-1).expand(-1, -1, u.size(-1))
    if reverse:
        v = u.new(u.size()).scatter_(1, order, u)
    else:
        v = u.gather(1, order)
    v = v.reshape(v.size()[:2] + x.size()[2:])
    return v


@cache
def fractal_order(N):
    """Binary search tree order.

    Args:
        N (int): Length of the sequence.

    Returns:
        list: Order.
    """
    a = deque([range(N - 1)])
    ret = [N - 1]
    i = 0
    while a:
        b = a.popleft()

        if b.stop <= b.start:
            continue

        h = (b.stop + b.start) // 2
        ret.append(h)

        if i % 2 == 0:
            a.append(range(h + 1, b.stop))
            a.append(range(b.start, h))
        else:
            a.append(range(b.start, h))
            a.append(range(h + 1, b.stop))

        i += 1

    return ret


def _shuffle(x, prompt_len, order_type="random"):
    if order_type == "random":
        order = torch.rand(x.size(), device=x.device)
        order[:, :prompt_len] = torch.arange(-prompt_len, 0, device=x.device)
        order = order.sort(1).indices
    elif order_type == "fractal":
        prompt = torch.arange(prompt_len, device=x.device)
        b = torch.tensor(fractal_order(x.size(1) - prompt_len), device=x.device)
        order = torch.cat([prompt, b + prompt_len])
        order = order.unsqueeze(0).expand(x.size(0), -1)
    else:
        order = (
            torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        )
    return order


def shuffle(x, prompt_len, order_type="random"):
    """Shuffle the sequence.

    Args:
        x (torch.Tensor): Input tensor.
        prompt_len (int): Length of the prompt (not shuffled).
        order_type (str): Type of order (left-to-right/ransom/fractal).

    Returns:
        torch.Tensor: Shuffled tensor.
    """
    order = _shuffle(x, prompt_len, order_type)
    return reorder(x, order), order
