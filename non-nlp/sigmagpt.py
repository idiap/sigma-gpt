# Copyright © <2024> Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Francois Fleuret <francois@fleuret.org>
# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>
# SPDX-License-Identifier: AGPL-3.0-only
"""Defines the σ-GPT model."""
import math

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from import_picoclvr import mygpt
from order import reorder, shuffle
from rejection_sampling import sample_and_evaluate


class AddDoublePositionalEncoding(nn.Module):
    """Add double positional encodings to the input.

    Paper: Vaswani et al. (2018) Attention is All You Need.
    PE_{t,2i} = sin(t/(L^{2i/D})), PE_{t,2i+1} = cos(t/(L^{2i/D}))
    Add a position encoding:
        - for the current token
        - for the next token to be generated

    These position encodings are concatenated together and added to the input.
    """

    def __init__(self, len_max):
        """Initialize the AddDoublePositionalEncoding module."""
        super().__init__()
        self.len_max = len_max

    def forward(self, bs, order):  # NxTxD, T
        """Forward pass of the AddDoublePositionalEncoding module.

        Args:
            bs (BracketedSequence): The input tensor.
            order (torch.Tensor): The order tensor.

        Returns:
            BracketedSequence: The output tensor.
        """
        if bs.first == 0:
            t = (
                torch.arange(bs.x.size(1) + 1, dtype=bs.x.dtype, device=bs.x.device)[
                    :, None
                ]
                - 1
            )
            j = torch.arange(bs.x.size(2) // 2, dtype=bs.x.dtype, device=bs.x.device)[
                None, :
            ]
            k = j % 2
            self.pe = (
                torch.sin(
                    t / (self.len_max ** ((j - k) / bs.x.size(2))) + math.pi / 2 * k
                )
                .unsqueeze(0)
                .expand(bs.x.size(0), -1, -1)
            )
            self.cache_y = bs.x.new(bs.x.size())

        order_output = order + 1
        order_input = F.pad(order + 1, (1, -1))

        pe_input = self.pe.gather(
            1, order_input.unsqueeze(-1).expand(-1, -1, self.pe.size(-1))
        )
        pe_output = self.pe.gather(
            1, order_output.unsqueeze(-1).expand(-1, -1, self.pe.size(-1))
        )

        pe = torch.cat((pe_input, pe_output), 2)

        self.cache_y[:, bs.first : bs.first + bs.nb] = (
            bs.slice() + pe[:, bs.first : bs.first + bs.nb]
        )

        bs.x = self.cache_y

        return bs


class QKVAttention(nn.Module):
    """QKV attention module."""

    def __init__(
        self,
        dim_in,
        dim_qk,
        dim_v,
        nb_heads=1,
        causal=False,
        attention_dropout=0.0,
        amm_generator=None,
    ):
        """Initialize the QKVAttention module."""
        super().__init__()

        def randw(*d):
            return nn.Parameter(torch.randn(*d) / math.sqrt(d[-1]))

        if amm_generator is None:
            self.amm_generator = (
                lambda d: torch.arange(d)[:, None] < torch.arange(d)[None, :]
            )
        else:
            self.amm_generator = amm_generator

        self.causal = causal
        self.attention_dropout = attention_dropout

        self.w_q = randw(nb_heads, dim_qk, dim_in)
        self.w_k = randw(nb_heads, dim_qk, dim_in)
        self.w_v = randw(nb_heads, dim_v, dim_in)
        self.w_o = randw(dim_v * nb_heads, dim_in)

        self.mode_burst = False

    def set_burst(self, burst):
        """If the model is in burst mode, all the tokens are considered as the next."""
        self.mode_burst = burst

    def burst(self, bs_q):
        """Burst mode for the QKVAttention module.

        Rely on the current KV cache to compute the attention.
        As it is carved, we can decompose it in block matrices and
        compute it directly from the cache
        No need to materialize the full matrix.

        See the appendix
        https://arxiv.org/abs/2404.09562

        Args:
            bs_q (BracketedSequence): The input tensor.

        Returns:
            BracketedSequence: The output tensor.
        """
        x_q = bs_q.x

        k1 = self.cache_k[:, :, : bs_q.first]
        v1 = self.cache_v[:, :, : bs_q.first]

        q = torch.einsum(
            "ntc,hdc->nhtd", x_q[:, bs_q.first : bs_q.first + bs_q.nb], self.w_q
        )
        k = torch.einsum(
            "ntc,hdc->nhtd", x_q[:, bs_q.first : bs_q.first + bs_q.nb], self.w_k
        )
        v = torch.einsum(
            "ntc,hdc->nhtd", x_q[:, bs_q.first : bs_q.first + bs_q.nb], self.w_v
        )
        att1 = q @ k1.transpose(-2, -1)
        att2 = (q * k).sum(-1, keepdim=True)
        att = torch.cat((att1, att2), -1) / math.sqrt(self.w_q.size(1))
        att = att.softmax(dim=-1)
        att = F.dropout(att, self.attention_dropout, self.training)

        att1 = att[..., :-1]
        att2 = att[..., [-1]]

        y1 = att1 @ v1 + att2 * v
        y1 = rearrange(y1, "b h t e -> b t (h e)")

        self.cache_y[:, bs_q.first : bs_q.first + bs_q.nb] = y1 @ self.w_o

        bs_q.x = self.cache_y

        return bs_q

    def forward(self, bs_q):
        """Forward pass of the QKVAttention module, standard QKV attention.

        Args:
            bs_q (BracketedSequence): The input tensor.

        Returns:
            BracketedSequence: The output tensor.
        """
        x_q = bs_q.x

        if bs_q.first == 0:
            self.cache_k = x_q.new_zeros(
                x_q.size(0), self.w_k.size(0), x_q.size(1), self.w_k.size(1)
            )
            self.cache_v = x_q.new_zeros(
                x_q.size(0), self.w_v.size(0), x_q.size(1), self.w_v.size(1)
            )
            self.cache_y = x_q.new_zeros(x_q.size(0), x_q.size(1), self.w_o.size(1))

        if self.mode_burst:
            return self.burst(bs_q)

        q = torch.einsum(
            "ntc,hdc->nhtd", x_q[:, bs_q.first : bs_q.first + bs_q.nb], self.w_q
        )
        self.cache_k[:, :, bs_q.first : bs_q.first + bs_q.nb] = torch.einsum(
            "ntc,hdc->nhtd", x_q[:, bs_q.first : bs_q.first + bs_q.nb], self.w_k
        )
        self.cache_v[:, :, bs_q.first : bs_q.first + bs_q.nb] = torch.einsum(
            "ntc,hdc->nhtd", x_q[:, bs_q.first : bs_q.first + bs_q.nb], self.w_v
        )

        a = torch.einsum(
            "nhtd,nhsd->nhts", q, self.cache_k[:, :, : bs_q.first + bs_q.nb]
        ) / math.sqrt(self.w_q.size(1))

        if self.causal:
            if bs_q.first == 0:
                self.cache_attzero = self.amm_generator(x_q.size(1)).to(q.device)[
                    None, None, :, :
                ]
            a = a.masked_fill(
                self.cache_attzero[
                    :, :, bs_q.first : bs_q.first + bs_q.nb, : bs_q.first + bs_q.nb
                ],
                float("-inf"),
            )

        a = a.softmax(dim=3)

        a = F.dropout(a, self.attention_dropout, self.training)

        y = torch.einsum(
            "nhts,nhsd->nthd", a, self.cache_v[:, :, : bs_q.first + bs_q.nb]
        ).flatten(2)

        self.cache_y[:, bs_q.first : bs_q.first + bs_q.nb] = y @ self.w_o

        bs_q.x = self.cache_y

        return bs_q


##############################


class SigmaGPT(nn.Module):
    """SigmaGPT model."""

    def __init__(
        self,
        vocabulary_size,
        dim_model,
        dim_keys,
        dim_hidden,
        nb_heads,
        nb_blocks,
        causal=False,
        dropout=0.0,
        len_max=1e5,
        amm_generator=None,
    ):
        """Initialize the SigmaGPT model."""
        super().__init__()

        assert dim_model % nb_heads == 0

        self.embedding = mygpt.CacheWrapper(
            nn.Embedding(vocabulary_size, dim_model), nn.Dropout(dropout)
        )
        self.pe = AddDoublePositionalEncoding(len_max)

        self.resetable = False
        self.burst = False

        trunk_blocks = []

        for _ in range(nb_blocks):
            trunk_blocks += [
                mygpt.WithResidual(
                    mygpt.CacheWrapper(nn.LayerNorm((dim_model,))),
                    QKVAttention(
                        dim_in=dim_model,
                        dim_qk=dim_keys,
                        dim_v=dim_model // nb_heads,
                        nb_heads=nb_heads,
                        causal=causal,
                        attention_dropout=dropout,
                        amm_generator=amm_generator,
                    ),
                ),
                mygpt.WithResidual(
                    mygpt.CacheWrapper(
                        nn.LayerNorm((dim_model,)),
                        nn.Linear(in_features=dim_model, out_features=dim_hidden),
                        nn.ReLU(),
                        nn.Linear(in_features=dim_hidden, out_features=dim_model),
                        nn.Dropout(dropout),
                    ),
                ),
            ]

        self.trunk = nn.Sequential(*trunk_blocks)

        self.readout = mygpt.CacheWrapper(
            nn.Linear(in_features=dim_model, out_features=vocabulary_size)
        )

        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Embedding):
                    m.weight.normal_(mean=0, std=2e-2)
                elif isinstance(m, nn.LayerNorm):
                    m.bias.zero_()
                    m.weight.fill_(1.0)

    def set_burst(self, burst):
        """Set the burst mode on the QKV modules."""
        if burst == self.burst:
            return
        self.burst = burst
        for m in self.modules():
            if isinstance(m, QKVAttention):
                m.set_burst(burst)

    def cache_ys(self):
        """Return the cache_y tensors."""
        for m in self.modules():
            if isinstance(
                m, (mygpt.CacheWrapper, AddDoublePositionalEncoding, QKVAttention)
            ) and hasattr(m, "cache_y"):
                yield m.cache_y

    def cache_pe(self):
        """Return the pe tensors."""
        for m in self.modules():
            if isinstance(m, AddDoublePositionalEncoding) and hasattr(m, "pe"):
                yield m.pe

    def cache_ks(self):
        """Return the cache_k tensors."""
        for m in self.modules():
            if isinstance(m, QKVAttention) and hasattr(m, "cache_k"):
                yield m.cache_k

    def cache_vs(self):
        """Return the cache_v tensors."""
        for m in self.modules():
            if isinstance(m, QKVAttention) and hasattr(m, "cache_v"):
                yield m.cache_v

    def all_caches(self):
        """Return all the caches."""
        return (
            list(self.cache_ys())
            + list(self.cache_pe())
            + list(self.cache_ks())
            + list(self.cache_vs())
        )

    def repeat_interleave_cache(self, n):
        """Repeat the cache n times."""
        if self.resetable:
            return

        for c in self.all_caches():
            c = torch.repeat_interleave(c, n, 0)
        self.resetable = True

    def select_sample_from_cache(self, index):
        """Select the samples from the cache.

        Typically used to remove the samples that have been fully generated.
        """
        for c in self.all_caches():
            c = c[index]

    def reshape_cache_and_select(self, b, n, amax):
        """Reshape the cache and select the samples.

        Typically used to only keep the final order after evaluation.
        """
        if not self.resetable:
            return

        for c in list(self.cache_ys()) + list(self.cache_pe()):
            c = c.reshape(b, n, -1, c.size(-1))
            c = c.take_along_dim(amax[..., None, None], 1).squeeze(1)

        for c in list(self.cache_ks()) + list(self.cache_vs()):
            _, h, t, e = c.size()
            c = c.reshape(b, n, h, t, e)
            c = c.take_along_dim(amax[..., None, None, None], 1).squeeze(1)
        self.resetable = False

    def forward(self, bs, mode="standard", order=None, burst=False):
        """Forward pass of the σ-GPT model."""
        bs = mygpt.BracketedSequence(F.pad(bs.x, (1, -1)), bs.first, bs.nb)

        if order is None:
            order = torch.arange(bs.x.size(1), device=bs.x.device)[None, :].expand_as(
                bs.x
            )
        bs = self.embedding(bs)
        bs = self.pe(bs, order)

        self.set_burst(burst)

        if mode == "standard":
            bs = self.trunk(bs)
            bs = self.readout(bs)
        elif mode == "head":
            bs = self.trunk(bs)

        self.set_burst(False)
        return bs

    def compute_loss(self, input, prompt_len, order_type="random"):
        """Compute the loss of the model depending on the order type."""
        if order_type == "left-to-right":
            output = self(mygpt.BracketedSequence(input)).x
        else:
            x, order = shuffle(input, prompt_len)
            output = self(mygpt.BracketedSequence(x), order=order).x
            output = reorder(output, order, reverse=True)
        return F.cross_entropy(output.transpose(1, 2), input)

    @torch.no_grad()
    def burst_sampling(self, input, ar_mask, norders=5, verbose=False):
        """Burst sampling."""
        to_generate = (ar_mask.sum(0) > 0).nonzero()
        prompt_len = to_generate.min()
        # if verbose:
        # print("Prompt length", prompt_len)

        x, order = shuffle(input, prompt_len)
        i = (ar_mask.sum(0) > 0).nonzero()
        self(mygpt.BracketedSequence(x, 0, i.min()), order=order)

        shuffled_sampled = x.clone()
        shuffled_sampled[:, prompt_len:] = 0

        # if verbose:
        #     print("First pass done")
        #     save(
        #         test_input,
        #         shuffled_sampled,
        #         order,
        #         ar_mask,
        #         dest / f"step000_{args.n_mazes}.png",
        #     )
        new_orders = order.clone()

        actives = torch.ones(input.shape[0], device=input.device, dtype=torch.bool)

        i = 1
        while True:
            try:
                shuffled_sampled_i, ar_mask_i, new_orders_i = (
                    shuffled_sampled[actives],
                    ar_mask[actives],
                    new_orders[actives],
                )
                shuffled_sampled_i, new_orders_i, ar_mask_i, done = sample_and_evaluate(
                    self,
                    shuffled_sampled_i,
                    ar_mask_i,
                    new_orders_i,
                    norders=norders,
                )
                shuffled_sampled[actives], ar_mask[actives], new_orders[actives] = (
                    shuffled_sampled_i,
                    ar_mask_i,
                    new_orders_i,
                )

                if not done.all():
                    not_done = ~done
                    actives[actives.clone()] = not_done
                    self.select_sample_from_cache(not_done)

            except ValueError:
                if verbose:
                    print(f"Regression Finished {i}")
                break

            # if verbose:
            #     save(
            #         test_input,
            #         shuffled_sampled,
            #         new_orders,
            #         dest / f"step{i:03d}_{args.n_mazes}.png",
            #     )
            #     print(f"Step {i} done")
            i += 1

        result = reorder(shuffled_sampled, new_orders, reverse=True)
        input.copy_(result)
        return i

    def shuffled_autoregression(
        self,
        input,
        ar_mask,
        order_type,
        deterministic_synthesis=False,
        forbidden_tokens=None,
        forced_biases=None,
    ):
        """Shuffled autoregression, like standard autoregression but with a random sequence."""
        to_generate = (ar_mask.sum(0) > 0).nonzero()
        prompt_len = to_generate.min()
        x, order = shuffle(input, prompt_len, order_type)
        if to_generate.min() > 0:
            # Needed to initialize the model's cache
            self(mygpt.BracketedSequence(x, 0, to_generate.min()), order=order)

        for s in range(to_generate.min(), to_generate.max() + 1):
            output = self(mygpt.BracketedSequence(x, s, 1), order=order).x
            logits = output[:, s]
            if forbidden_tokens is not None:
                logits = logits.masked_fill(forbidden_tokens, float("-inf"))
            if forced_biases is not None:
                logits = logits + forced_biases[None, :]
            if deterministic_synthesis:
                t_next = logits.argmax(1)
            else:
                dist = torch.distributions.categorical.Categorical(logits=logits)
                t_next = dist.sample()
            x[:, s] = ar_mask[:, s] * t_next + (1 - ar_mask[:, s]) * x[:, s]

        result = reorder(x, order, reverse=True)
        input.copy_(result)

    masked_inplace_autoregression = mygpt.MyGPT.masked_inplace_autoregression
