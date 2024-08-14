# Copyright © <2024> Idiap Research Institute <contact@idiap.ch>

# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>

# SPDX-License-Identifier: AGPL-3.0-only
import math

import pytest
import torch
import torch.nn.functional as F

import sigmagpt
from import_nanogpt import model

devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda")

L, T, H, E = 0, 100, 8, 64
config = model.GPTConfig(
    vocab_size=2,
    block_size=256,
    n_layer=1,
    n_head=H,
    n_embd=E,
    dropout=0.0,
)

self_attention = sigmagpt.CausalSelfAttention(config)


@pytest.mark.parametrize("device", devices)
def test_flash_attention_no_cache_no_dropout(device):
    self_attention.to(device)

    q = torch.randn(10, H, T, E // H, device=device)
    k = torch.randn(10, H, T, E // H, device=device)
    v = torch.randn(10, H, T, E // H, device=device)

    y = self_attention.manual_self_attention(q, k, v, L, T)
    assert y.shape == (10, H, T, E // H)

    y_reg = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=True,
    )

    assert torch.allclose(y, y_reg, atol=1e-5)


@pytest.mark.parametrize("device", devices)
def test_attn_dropout_is_identity_during_eval(device):
    self_attention.to(device)
    self_attention.eval()

    att = torch.randn(10, H, T, T, device=device)
    y = self_attention.attn_dropout(att)
    assert torch.allclose(y, att, atol=1e-5)


@pytest.mark.parametrize("device", devices)
def test_attn_dropout_is_not_identity_during_training(device):
    T, H, E = 100, 8, 64
    config = model.GPTConfig(
        vocab_size=2,
        block_size=256,
        n_layer=1,
        n_head=H,
        n_embd=E,
        dropout=0.99,
    )

    self_attention = sigmagpt.CausalSelfAttention(config)
    self_attention.to(device)
    self_attention.train()

    att = torch.randn(10, H, T, T, device=device)
    y = self_attention.attn_dropout(att)
    assert not torch.allclose(y, att, atol=1e-5)


@pytest.mark.parametrize("device", devices)
def test_flash_attention_no_cache_dropout_during_eval(device):
    L, T, H, E = 0, 100, 8, 64
    config = model.GPTConfig(
        vocab_size=2,
        block_size=256,
        n_layer=1,
        n_head=H,
        n_embd=E,
        dropout=0.1,
    )

    self_attention = sigmagpt.CausalSelfAttention(config)
    self_attention.to(device)
    self_attention.eval()

    q = torch.randn(10, H, T, E // H, device=device)
    k = torch.randn(10, H, T, E // H, device=device)
    v = torch.randn(10, H, T, E // H, device=device)

    y = self_attention.manual_self_attention(q, k, v, L, T)
    assert y.shape == (10, H, T, E // H)

    y_reg = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0,
        is_causal=True,
    )

    assert torch.allclose(y, y_reg, atol=1e-5)


@pytest.mark.parametrize("device", devices)
def test_flash_attention_with_cache_should_fail(device):
    B, L, T, H, E = 10, 100, 100, 8, 64
    config = model.GPTConfig(
        vocab_size=2,
        block_size=256,
        n_layer=1,
        n_head=H,
        n_embd=E,
        dropout=0.1,
    )

    self_attention = sigmagpt.CausalSelfAttention(config)
    self_attention.to(device)
    self_attention.train()

    q = torch.randn(B, H, T, E // H, device=device)
    k1 = torch.randn(B, H, T, E // H, device=device)
    v1 = torch.randn(B, H, T, E // H, device=device)
    k2 = torch.randn(B, H, T, E // H, device=device)
    v2 = torch.randn(B, H, T, E // H, device=device)

    KeyValues = sigmagpt.KeysValues(B, H, 256, E, 1, device=device)
    kv = KeyValues[0]

    kv.update(k1, v1)
    kv.update(k2, v2)
    k, v = kv.get()

    assert k.shape == (B, H, 2 * T, E // H)
    assert v.shape == (B, H, 2 * T, E // H)

    y = self_attention.manual_self_attention(q, k, v, L, T)
    assert y.shape == (B, H, T, E // H)

    y_reg = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=True,
    )
    assert y_reg.shape == (B, H, T, E // H)

    assert not torch.allclose(y, y_reg, atol=1e-5)


@pytest.mark.parametrize("device", devices)
def test_flash_attention_with_cache_use_another_masking(device):
    B, L, T, H, E = 10, 100, 100, 8, 64
    config = model.GPTConfig(
        vocab_size=2,
        block_size=256,
        n_layer=1,
        n_head=H,
        n_embd=E,
        dropout=0.1,
    )

    self_attention = sigmagpt.CausalSelfAttention(config)
    self_attention.to(device)
    self_attention.train()

    q = torch.randn(B, H, T, E // H, device=device)
    k1 = torch.randn(B, H, T, E // H, device=device)
    v1 = torch.randn(B, H, T, E // H, device=device)
    k2 = torch.randn(B, H, T, E // H, device=device)
    v2 = torch.randn(B, H, T, E // H, device=device)

    KeyValues = sigmagpt.KeysValues(B, H, 256, E, 1, device=device)
    kv = KeyValues[0]

    kv.update(k1, v1)
    kv.update(k2, v2)
    k, v = kv.get()

    assert k.shape == (B, H, 2 * T, E // H)
    assert v.shape == (B, H, 2 * T, E // H)

    bias = torch.tril(torch.ones(256, 256, device=device))

    # https://pytorch.org/docs/2.4/generated/torch.nn.functional.scaled_dot_product_attention.html#torch-nn-functional-scaled-dot-product-attention
    # Is causal is the problem

    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    # This should be L: L + T instead of 0:T
    att = att.masked_fill(bias[0:T, : L + T] == 0, float("-inf"))
    att = F.softmax(att, dim=-1)
    y = att @ v

    assert y.shape == (B, H, T, E // H)

    y_reg = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=True,
    )
    assert y_reg.shape == (B, H, T, E // H)

    assert torch.allclose(y, y_reg, atol=1e-5)
