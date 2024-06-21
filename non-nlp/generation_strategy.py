# Copyright © <2024> Idiap Research Institute <contact@idiap.ch>

# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>

# SPDX-License-Identifier: AGPL-3.0-only
"""Defines generation strategies for autoregressive models."""

from functools import partial

import torch
import tqdm

from import_picoclvr import tasks


def shuffled_autoregression(
    model,
    batch_size,
    input,
    ar_mask,
    deterministic_synthesis,
    device,
    forbidden_tokens=None,
    logit_biases=None,
    progress_bar_desc="shuffled_autoregression",
    order_type="random",
):
    """Do a standard autoregression in a random order.

    Args:
        model (torch.nn.Module): The model to generate with.
        batch_size (int): The batch size to use.
        input (torch.Tensor): The input tensor.
        ar_mask (torch.Tensor): Tell which token needs to be generated.
        deterministic_synthesis (bool): Whether to use deterministic synthesis.
        device (torch.device): The device to use.
        forbidden_tokens (torch.Tensor): The forbidden tokens.
        logit_biases (torch.Tensor): The logit biases.
        progress_bar_desc (str): The description for the progress bar.
        order_type (str): The order type to use.

    Returns:
        None (inplace operation)
    """
    assert input.size() == ar_mask.size()

    batches = zip(input.split(batch_size), ar_mask.split(batch_size))

    if progress_bar_desc is not None:
        batches = tqdm.tqdm(
            batches,
            dynamic_ncols=True,
            desc=progress_bar_desc,
            total=(input.size(0) + batch_size - 1) // batch_size,
        )

    with torch.autograd.no_grad():
        t = model.training
        model.eval()

        for input, ar_mask in batches:
            model.shuffled_autoregression(
                input,
                ar_mask,
                order_type,
                deterministic_synthesis,
                forbidden_tokens,
                logit_biases,
            )

        model.train(t)


def burst_sampling(
    model,
    batch_size,
    input,
    ar_mask,
    deterministic_synthesis,
    device,
    forbidden_tokens=None,
    logit_biases=None,
    progress_bar_desc="burst",
    norders=5,
    verbose=False,
):
    """Do a burst sampling autoregression.

    Algorithm of the method is given in the paper:
    https://arxiv.org/abs/2404.09562

    Args:
        model (torch.nn.Module): The model to generate with.
        batch_size (int): The batch size to use.
        input (torch.Tensor): The input tensor.
        ar_mask (torch.Tensor): Tell which token needs to be generated.
        deterministic_synthesis (bool): Whether to use deterministic synthesis.
        device (torch.device): The device to use.
        forbidden_tokens (torch.Tensor): The forbidden tokens.
        logit_biases (torch.Tensor): The logit biases.
        progress_bar_desc (str): The description for the progress bar.
        norders (int): The number of orders to use.
        verbose (bool): Whether to print the number of steps.

    Returns:
        None (inplace operation)
    """
    assert input.size() == ar_mask.size()

    batches = zip(input.split(batch_size), ar_mask.split(batch_size))

    if progress_bar_desc is not None:
        batches = tqdm.tqdm(
            batches,
            dynamic_ncols=True,
            desc=progress_bar_desc,
            total=(input.size(0) + batch_size - 1) // batch_size,
        )

    with torch.autograd.no_grad():
        t = model.training
        model.eval()

        steps = []
        for input, ar_mask in batches:
            s = model.burst_sampling(
                input,
                ar_mask,
                norders=norders,
                verbose=verbose,
            )
            steps.append(s)

        print(f"Mean Burst Steps: {sum(steps)/len(steps):.1f}, {steps}")

        model.train(t)


standard_autoregression = tasks.masked_inplace_autoregression
fractal_autoregression = partial(
    shuffled_autoregression,
    order_type="fractal",
    progress_bar_desc="fractal_autoregression",
)


generate_strategies = {
    "left-to-right": standard_autoregression,
    "shuffle": shuffled_autoregression,
    "fractal": fractal_autoregression,
    "burst": burst_sampling,
}
