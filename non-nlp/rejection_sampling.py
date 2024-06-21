# Copyright © <2024> Idiap Research Institute <contact@idiap.ch>

# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>

# SPDX-License-Identifier: AGPL-3.0-only
"""Defines parts of the rejection sampling strategy for σ-GPT."""
import torch

from import_picoclvr import mygpt
from order import reorder, shuffle


@torch.no_grad()
def sample_remaining(model, input, ar_mask, order, start_i, norders):
    """Sample the remaining tokens as if they were the next tokens to be generated.

    Args:
        model (torch.nn.Module): The model to generate with.
        input (torch.Tensor): The input tensor.
        ar_mask (torch.Tensor): Tell which token needs to be generated.
        order (torch.Tensor): The order tensor.
        start_i (int): The index where to start sampling.
        norders (int): The number of orders that the sequences should be repeated.
    """
    # Here we use the burst parameter, which leverage the cache
    o = model(
        mygpt.BracketedSequence(input, start_i, input.shape[1]), order=order, burst=True
    )
    p = o.x.softmax(-1)

    # We repeat the sequences norders times
    p = p.repeat_interleave(norders, 0)
    mask = ar_mask.repeat_interleave(norders, 0)
    inp = input.repeat_interleave(norders, 0)

    # We sample the remaining tokens
    dist = torch.distributions.categorical.Categorical(probs=p)
    sampled = dist.sample()
    sampled = mask * sampled + (1 - mask) * inp
    return sampled, p


@torch.no_grad()
def sample_and_evaluate(model, input, ar_mask, order, norders=5):
    """Sample the remaining tokens as if they were the next tokens to be generated and evaluate them.

    The evaluation is done by sampling N different orders
    Computing the probability of the sampled tokens under the order
    And keeping the order which validates the most the sampled tokens.

    Args:
        model (torch.nn.Module): The model to generate with.
        input (torch.Tensor): The input tensor.
        ar_mask (torch.Tensor): Tell which token needs to be generated.
        order (torch.Tensor): The order tensor.
        norders (int): The number of orders that the sequences should be repeated.

    Returns:
        torch.Tensor: The sampled tokens.
        torch.Tensor: The orders.
    """
    batch_size, seq_len = input.shape
    i = (ar_mask.sum(0) > 0).nonzero()
    if len(i) == 0:
        raise ValueError("Regression Finished")
    i = i.min()

    sampled, probs = sample_remaining(model, input, ar_mask, order, i, norders)
    order = order.repeat_interleave(norders, 0)

    probs = reorder(probs, order, reverse=True)
    sampled = reorder(sampled, order, reverse=True)

    confs = torch.gather(probs, 2, sampled.unsqueeze(2)).squeeze(2)

    orders, _ = shuffle(order, i)

    shuffled_sampled = reorder(sampled, orders)

    # Repeating the cache norders times
    model.repeat_interleave_cache(norders)
    # We now evaluate the sampled tokens under the different orders
    q = model(
        mygpt.BracketedSequence(shuffled_sampled, i, shuffled_sampled.size(1)),
        order=orders,
    )

    qs = q.x.softmax(2)
    # Keeping the probability of the sampled tokens under the different orders
    qs = qs.gather(2, shuffled_sampled.unsqueeze(2)).squeeze(2)
    qs = qs.reshape(batch_size, norders, seq_len)

    # We now compute the acceptance ratio of the rejection sampling scheme
    u = torch.rand_like(confs)
    u = u.reshape(batch_size, norders, seq_len)

    # Putting the acceptance ratio in the right order
    confs_in_q_order = reorder(confs, orders)
    confs_in_q_order = confs_in_q_order.reshape(batch_size, norders, seq_len)

    rs = qs / confs_in_q_order
    accepts = (u < rs).float()

    done = (accepts == 1).all(2)

    # Selecting the order which validates the most the sampled tokens
    idx = torch.where(done, seq_len, accepts.argmin(2))
    done = done.any(1)

    max, amax = idx.max(1, keepdim=True)
    min = max.min()
    # Reset the cache to its right size (one order)
    model.reshape_cache_and_select(batch_size, norders, amax)

    orders = orders.reshape(batch_size, norders, seq_len)
    new_orders = orders.take_along_dim(amax.unsqueeze(2), 1).squeeze(1)

    ar_mask[done] = 0
    ar_mask[~done, :min] = 0

    shuffled_sampled = shuffled_sampled.reshape(batch_size, norders, seq_len)
    shuffled_sampled = shuffled_sampled.take_along_dim(amax.unsqueeze(2), 1).squeeze(1)

    return shuffled_sampled, new_orders, ar_mask, done
