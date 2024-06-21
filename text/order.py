# Copyright © <2024> Idiap Research Institute <contact@idiap.ch>

# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>

# SPDX-License-Identifier: AGPL-3.0-only
"""Ordering functions for σ-GPT models."""

from collections import deque
from functools import cache

import torch


@cache
def fractal_order(N):
    """Binary search tree order.

    Args:
        N (int): Length of the sequence.

    Returns:
        list: Order.
    """
    a = deque([range(N)])
    ret = []
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


def order(x, prompt_len, order_type="left-to-right"):
    """Get a permutation for the sequence.

    Args:
        x (torch.Tensor): Input sequence.
        prompt_len (int): Length of the prompt.
        order_type (str): Type of permutation (left-to-right).



    """
    if order_type == "random":
        order = torch.rand(x.size(), device=x.device)
        order[:, :prompt_len] = torch.arange(-prompt_len, 0, device=x.device)
        return order.sort(1).indices
    elif order_type == "fractal":
        prompt = torch.arange(prompt_len, device=x.device)
        b = torch.tensor(fractal_order(x.size(1) - prompt_len), device=x.device)
        order = torch.cat([prompt, b + prompt_len])
        return order.unsqueeze(0).expand(x.size(0), -1)
    elif order_type == "left-to-right":
        return (
            torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        )
    else:
        raise ValueError("Unknown order type")


def order_mask(mask, order_type="left-to-right"):
    """Get a permutation for the sequence given a mask.

    The mask corresponds to the elements that should be predicted.

    Args:
        mask (torch.Tensor): Mask.
        order_type (str): Type of permutation (left-to-right/random).
    """
    if order_type not in ["left-to-right", "random"]:
        raise ValueError(f"order_mask not implemented for order type {order_type}")

    if order_type == "random":
        m = mask * 2 - 1
        o = torch.rand(m.size(), device=m.device)
        o = o * m
        return o.sort(1).indices

    # Else everything should be at the beginning
    prompt_len = (1 - mask).sum(1).min().item()  # to predict = 1
    return order(mask, prompt_len, order_type)


def shuffle(x, prompt_len, order_type="left-to-right"):
    """Shuffle the sequence.

    Args:
        x (torch.Tensor): Input tensor.
        prompt_len (int): Length of the prompt (not shuffled).
        order_type (str): Type of order (left-to-right/random/fractal).

    Returns:
        torch.Tensor: Shuffled tensor.
    """
    o = order(x, prompt_len, order_type)
    return reorder(x, o), o


def shuffle_after_prompt(prompt_len, i, L):
    """Shuffle the sequence after the prompt.

    Args:
        prompt_len (int): Length of the prompt.
        i (int): Position of the prompt.
        L (int): Length of the sequence.

    Returns:
        torch.Tensor: Order.
    """
    p = torch.randperm(prompt_len)
    s = torch.randperm(i - prompt_len) + prompt_len
    e = torch.randperm(L - i) + i
    return torch.cat([p, s, e])[None, :]


def reorder(x, order, reverse=False):  # x is NxTxD1x...xDk, order is NxT'
    """Reorder the sequence.

    Args:
        x (torch.Tensor): Input tensor.
        order (torch.Tensor): Order.
        reverse (bool): Reverse the order.

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


def apply_order(x, o):
    """Apply the order to the sequence.

    Args:
        x (torch.Tensor): Input tensor.
        o (torch.Tensor): Order tensor.

    Returns:
        torch.Tensor: Ordered tensor.
    """
    y = x.clone()
    x = x.gather(1, o)
    x = x[:, :-1]
    x = x.contiguous()

    y = y.gather(1, o)
    y = y[:, 1:]
    y = y.contiguous()
    return x, y
