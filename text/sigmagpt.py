# Copyright © <2024> Idiap Research Institute <contact@idiap.ch>

# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>

# SPDX-License-Identifier: AGPL-3.0-only
"""Adapt nanoGPT model (nanoGPT/model.py) to σ-GPT."""
import math
import re
import sys

import torch
from torch import nn
from torch.nn import functional as F

from caching import KeysValues as original_KeysValues
from caching import KVCache as original_KVCache
from import_nanogpt import model
from order import order_mask, reorder


class KVCache(original_KVCache):
    """Override the KVCache class to add rejection sampling methods."""

    @property
    def device(self):
        """Return the device of the KVCache."""
        return self._k_cache._cache.device

    def clone(self) -> "KVCache":
        """Clone the KVCache for evaluation."""
        n, num_heads, max_tokens, embed_dim = self.shape
        kv = KVCache(n, num_heads, max_tokens, embed_dim * num_heads, self.device)
        kv._k_cache._cache = self._k_cache._cache.clone()
        kv._v_cache._cache = self._v_cache._cache.clone()

        kv._k_cache._size = self._k_cache._size
        kv._v_cache._size = self._v_cache._size
        return kv

    def repeat_interleave(self, L) -> "KVCache":
        """Repeat the KVCache for evaluation under n orders."""
        n, num_heads, max_tokens, embed_dim = self.shape
        kv = KVCache(n * L, num_heads, max_tokens, embed_dim * num_heads, self.device)

        kv._k_cache._cache = self._k_cache._cache.repeat_interleave(L, dim=0)
        kv._v_cache._cache = self._v_cache._cache.repeat_interleave(L, dim=0)

        kv._k_cache._size = self._k_cache._size
        kv._v_cache._size = self._v_cache._size

        return kv

    def reshape_take_and_slice(self, shape, idx, dim, min) -> "KVCache":
        """Reshape, take and slice the KVCache to take the best sequence."""
        _, num_heads, max_tokens, embed_dim = self.shape

        kv = KVCache(
            shape[0], num_heads, max_tokens, embed_dim * num_heads, self.device
        )

        kv._k_cache._cache[:, :, :min, :] = (
            self._k_cache._cache[:, :, :min, :]
            .reshape(shape)
            .take_along_dim(idx, dim)
            .squeeze(1)[:, :, :min, :]
            .clone()
        )
        kv._v_cache._cache[:, :, :min, :] = (
            self._v_cache._cache[:, :, :min, :]
            .reshape(shape)
            .take_along_dim(idx, dim)
            .squeeze(1)[:, :, :min, :]
            .clone()
        )

        kv._k_cache._size = min.item()
        kv._v_cache._size = min.item()

        return kv


class KeysValues(original_KeysValues):
    """Override the KeysValues class to add rejection sampling methods."""

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

    @property
    def shape(self):
        """Return the shape of the KeysValues."""
        return self._keys_values[0].shape

    @property
    def size(self):
        """Return the size of the KeysValues."""
        return self.shape[2]

    @property
    def device(self):
        """Return the device of the KeysValues."""
        return self._keys_values[0].device

    def clone(self):
        """Clone the KeysValues for evaluation."""
        n, num_heads, max_tokens, embed_dim = self._keys_values[0].shape
        num_layers = len(self)
        kv = KeysValues(
            n, num_heads, max_tokens, embed_dim * num_heads, num_layers, self.device
        )
        kv._keys_values = tuple(kv_cache.clone() for kv_cache in self._keys_values)
        return kv

    def repeat_interleave(self, n):
        """Repeat the KeysValues for evaluation under n orders."""
        b, num_heads, max_tokens, embed_dim = self._keys_values[0].shape
        num_layers = len(self)
        kv = KeysValues(
            b * n, num_heads, max_tokens, embed_dim * num_heads, num_layers, self.device
        )
        kv._keys_values = tuple(
            kv_cache.repeat_interleave(n) for kv_cache in self._keys_values
        )
        return kv

    def reshape_take_and_slice(self, shape, idx, dim, min):
        """Reshape, take and slice the KeysValues to take the best sequence."""
        _, num_heads, max_tokens, embed_dim = self._keys_values[0].shape
        new_b = shape[0]
        num_layers = len(self)
        kv = KeysValues(
            new_b, num_heads, max_tokens, embed_dim * num_heads, num_layers, self.device
        )
        kv._keys_values = tuple(
            kv_cache.reshape_take_and_slice(shape, idx, dim, min)
            for kv_cache in self._keys_values
        )
        return kv


class CausalSelfAttention(model.CausalSelfAttention):
    """Override the CausalSelfAttention class to add burst mode and kv cache."""

    def __init__(self, config):
        """Initialize the CausalSelfAttention module."""
        super().__init__(config)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)),
            persistent=False,
        )

    def burst(self, kv_cache, q, k, v):
        """Burst mode for the QKVAttention module.

        Rely on the current KV cache to compute the attention.
        As it is carved, we can decompose it in block matrices and
        compute it directly from the cache
        No need to materialize the full matrix.

        See the appendix
        https://arxiv.org/abs/2404.09562

        Args:
            kv_cache (KVCache): KV cache.
            q (torch.Tensor): Query.
            k (torch.Tensor): Key.
            v (torch.Tensor): Value.

        Returns:
            torch.Tensor: Output.
        """
        k1, v1 = kv_cache.get()

        att1 = q @ k1.transpose(-2, -1)
        # no masking on kv cache, tokens can see them all

        att2 = (q * k).sum(-1, keepdim=True)
        att = torch.cat((att1, att2), dim=-1) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        att1 = att[..., :-1]
        att2 = att[..., [-1]]

        # Works even if cache is empty
        y = att1 @ v1 + att2 * v

        B, H, T, E = y.size()
        C = H * E

        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        y = self.resid_dropout(self.c_proj(y))
        return y

    def forward(self, x, kv_cache=None, burst=False):
        """Forward pass of the CausalSelfAttention module add burst mode and kvcache."""
        # batch size, sequence length, embedding dimensionality (n_embd)
        (B, T, C) = x.size()

        if kv_cache is not None:
            b, nh, L, c = kv_cache.shape
            assert nh == self.n_head and b == B and c * nh == C
        else:
            L = 0

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        if burst:
            y = self.burst(kv_cache, q, k, v)
            return y

        if kv_cache is not None:
            kv_cache.update(k, v)
            k, v = kv_cache.get()

        y = self.manual_self_attention(q, k, v, L, T)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
        # manual implementation of attention

    def manual_self_attention(self, q, k, v, L, T):
        """Manual implementation of the self-attention.

        Flash attention with is_causal=True fails with cache because it takes the wrong mask.

        when q is (B, H, 2, E) and k,v are longer (B, H, 5, E)
        the mask should be in that case

        [[1, 1, 1, 0, 0],
         [1, 1, 1, 1, 0],
         (bottom left of triangular matrix)

         but default takes
        [[1, 0, 0, 0, 0],
         [1, 1, 0, 0, 0],
        (upper left triangular matrix)

        A fix could be to use matrix but this has to be benchmarked and tested.

        Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        """
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[L : L + T, : L + T] == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        return att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)


class Block(nn.Module):
    """Override the Block class to add burst mode and kv cache."""

    def __init__(self, config):
        """Initialize the Block module."""
        super().__init__()
        self.ln_1 = model.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = model.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = model.MLP(config)

    def forward(self, x, kv_cache=None, burst=False):
        """Forward pass of the Block module."""
        x = x + self.attn(self.ln_1(x), kv_cache=kv_cache, burst=burst)
        x = x + self.mlp(self.ln_2(x))
        return x


class sigmaGPT(nn.Module):
    """Adapt GPT model to σ-GPT."""

    _init_weights = model.GPT._init_weights
    get_num_params = model.GPT.get_num_params
    configure_optimizers = model.GPT.configure_optimizers
    estimate_mfu = model.GPT.estimate_mfu

    def __init__(self, config):
        """Initialize the sigmaGPT model."""
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(
                    config.block_size, config.n_embd // 2
                ),  # Only change here // 2
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": model.LayerNorm(config.n_embd, bias=config.bias),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def _pos_emb(self, idx, order):
        t = idx.size(1)
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # Order should always be t + 1
        if order.size(1) > t + 1:
            # print("Bigger")
            order = order[:, : t + 1]

        order_input = order[:, :-1, None]
        order_target = order[:, 1:, None]

        order = torch.cat((order_input, order_target), dim=2)
        # print(order)

        return self.transformer.wpe(order).flatten(2)

    def forward(
        self, idx, order, targets=None, optimize=True, kv_cache=None, burst=False
    ):
        """Forward pass of the sigmaGPT model."""
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self._pos_emb(idx, order)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        for i, block in enumerate(self.transformer.h):
            x = block(x, None if kv_cache is None else kv_cache[i], burst)
        x = self.transformer.ln_f(x)

        if targets is None:
            if optimize:
                # inference-time mini-optimization: only forward the lm_head on the very last position
                # note: using list [-1] to preserve the time dim
                return self.lm_head(x[:, [-1], :]), None

            return self.lm_head(x), None

        logits = self.lm_head(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
        )
        return logits, loss

    @torch.no_grad()
    def sample_rem(self, prompt, order, rem, kv, temperature=1.0):
        """Sample the remaining tokens."""
        seq = prompt[:, [-1]].repeat(1, rem.shape[1])
        rem = torch.cat((order[:, [-1]], rem), 1)

        logits, _ = self(seq, rem, kv_cache=kv, optimize=False, burst=True)
        logits = logits / temperature
        p = F.softmax(logits, dim=2)
        dist = torch.distributions.Categorical(probs=p)
        return dist.sample(), p

    @torch.no_grad()
    def compute_qs(self, prompt, sampled, orders, perms, kv, norders, temperature=1.0):
        """Compute the acceptance probabilities."""
        kvc = kv.clone()
        kvc = kvc.repeat_interleave(norders)
        b = sampled.shape[0]
        sampled = sampled.repeat_interleave(norders, dim=0)
        pr = prompt[:, [-1]].repeat_interleave(norders, dim=0)

        seq = sampled.gather(1, perms)
        seq = torch.cat((pr, seq), dim=1)
        target = seq.clone()
        target = target[:, 1:]
        seq = seq[:, :-1]
        logits = self(seq, orders, kv_cache=kvc, optimize=False)[0]
        logits = logits / temperature
        q = logits.softmax(2)
        qs = torch.gather(q, 2, target.unsqueeze(2)).squeeze(2)
        qs = qs.reshape(b, norders, -1)
        return qs, kvc

    @torch.no_grad()
    def sample_and_evaluate(
        self, prompt, order, rem, kv, norders=3, max_tokens=1024, temperature=1.0
    ):
        """Sample and evaluate the tokens."""
        sampled, probs = self.sample_rem(
            prompt, order, rem, kv, temperature=temperature
        )

        p = torch.gather(probs, 2, sampled.unsqueeze(2)).squeeze(2)

        o_and_last_pos = torch.cat((order[:, [-1]], rem), dim=1)
        o_and_last_pos = o_and_last_pos.repeat_interleave(norders, dim=0)
        perms = torch.rand_like(o_and_last_pos.float())
        perms[:, 0] = -1
        perms = perms.sort(1).indices
        o_and_last_pos = o_and_last_pos.gather(1, perms)
        rem = rem.repeat_interleave(norders, dim=0)
        perms_without_last = perms[:, 1:] - 1

        qs, kvc = self.compute_qs(
            prompt,
            sampled,
            o_and_last_pos,
            perms_without_last,
            kv,
            norders,
            temperature,
        )

        rem = rem.gather(1, perms_without_last)
        perms_without_last = perms_without_last.reshape(sampled.shape[0], norders, -1)
        o_and_last_pos = o_and_last_pos.reshape(sampled.shape[0], norders, -1)
        rem = rem.reshape(sampled.shape[0], norders, -1)

        u = torch.rand_like(p)

        p_in_qs_order = torch.gather(
            p[:, None].repeat(1, norders, 1), 2, perms_without_last
        )
        u_in_qs_order = torch.gather(
            u[:, None].repeat(1, norders, 1), 2, perms_without_last
        )

        rs = qs / p_in_qs_order
        accepts = (u_in_qs_order < rs).float()

        idx = torch.where((accepts == 1).all(2), max_tokens, accepts.argmin(2))
        max, amax = idx.max(1, keepdim=True)
        min = max.min()

        new_order = rem.take_along_dim(amax.unsqueeze(2), 1).squeeze(1)
        new_rem = new_order[:, min:]
        new_order = new_order[:, :min]
        perms_without_last = perms_without_last.take_along_dim(
            amax.unsqueeze(2), 1
        ).squeeze(1)
        perms_without_last = perms_without_last[:, :min]
        new_prompt = sampled.gather(1, perms_without_last)

        B, num_heads, _, embed_dim = kvc._keys_values[0].shape

        new_kv_size = min + kv.size

        if min < max_tokens:
            kv = kvc.reshape_take_and_slice(
                (B // norders, norders, num_heads, new_kv_size, embed_dim),
                amax[:, :, None, None, None],
                1,
                new_kv_size,
            )

        order = torch.cat((order, new_order), dim=1)
        prompt = torch.cat((prompt, new_prompt), dim=1)

        return order, prompt, new_rem, kv

    @torch.no_grad()
    def generate_rejection_sampling(
        self,
        idx,
        mask_idx,
        norders=5,
        verbose=False,
        decode=None,
        temperature=1.0,
        **kwargs,
    ):
        """Generate a sequence with rejection sampling.

        Args:
            idx (torch.Tensor): Input tensor.
            mask_idx (torch.Tensor): Mask tensor containing prompt.
            norders (int): Number of orders.
            verbose (bool): Verbose mode.
            decode (function): Decode function.

        Returns:
            final (torch.Tensor): Final tensor.
            k (int): Number of iterations.
        """
        b, num_tokens = idx.shape
        o = order_mask(mask_idx, order_type="random")
        shuffled_idx = idx.gather(1, o).contiguous()

        should_print = verbose and decode is not None
        if should_print:
            init_prompt(num_tokens)

        to_generate = mask_idx.sum(1).max().item()
        prompt_len = num_tokens - to_generate

        kv_cache = self.generate_empty_keys_values(b, num_tokens)
        # Init kv cache
        self(
            shuffled_idx[:, : prompt_len - 1],
            order=o[:, :prompt_len],
            optimize=False,
            kv_cache=kv_cache,
        )

        partial_order = o[:, :prompt_len]
        prompt = shuffled_idx[:, :prompt_len]
        rem = o[:, prompt_len:]
        k = 0

        while True or k < num_tokens:
            k += 1
            partial_order, prompt, rem, kv_cache = self.sample_and_evaluate(
                prompt,
                partial_order,
                rem,
                kv_cache,
                norders=norders,
                max_tokens=num_tokens,
                temperature=temperature,
            )
            prompt_len = prompt.shape[1]
            shuffled_idx[:, :prompt_len] = prompt

            o = torch.cat((partial_order, rem), dim=1)

            if should_print:
                nn = reorder(shuffled_idx, o, reverse=True)[0].tolist()
                refresh_lines(decode(nn), k, kv_cache.size, prompt.shape[1])
            # print(order.shape)
            if partial_order.shape[1] >= num_tokens:
                break

        final = reorder(shuffled_idx, o, reverse=True)

        return final, k

    def generate_empty_keys_values(self, n, max_tokens):
        """Generate an empty KeysValues object."""
        device = next(self.parameters()).device
        return KeysValues(
            n,
            self.config.n_head,
            max_tokens,
            self.config.n_embd,
            self.config.n_layer,
            device,
        )

    @torch.no_grad()
    def generate_autoregressively_with_kvcache(
        self,
        idx,
        mask_idx,
        temperature=1.0,
        top_k=None,
        order_type=None,
        verbose=False,
        decode=None,
    ):
        """Generate a sequence autoregressively in specified order with kv cache.

        Args:
            idx (torch.Tensor): Input tensor.
            mask_idx (torch.Tensor): Mask tensor containing prompt.
            temperature (float): Temperature.
            top_k (int): Top k.
            order_type (str): Order type.
            verbose (bool): Verbose mode.
            decode (function): Decode function.

        Returns:
            final (torch.Tensor): Final tensor.
        """
        b, num_tokens = idx.shape
        o = order_mask(mask_idx, order_type=order_type)

        shuffled_idx = idx.gather(1, o).contiguous()

        should_print = verbose and decode is not None
        print(should_print)
        if should_print:
            init_prompt(num_tokens)

        to_generate = mask_idx.sum(1).max().item()
        prompt_len = num_tokens - to_generate + 1

        kv_cache = self.generate_empty_keys_values(b, num_tokens)
        # init

        for i in range(prompt_len, num_tokens):
            # forward the model to get the logits for the index in the sequence
            if i == prompt_len:
                logits, _ = self(
                    shuffled_idx[:, : i - 1],
                    order=o[:, :i],
                    optimize=False,
                    kv_cache=kv_cache,
                )
            else:
                logits, _ = self(
                    shuffled_idx[:, [i - 1]],
                    order=o[:, i - 1 : i + 1],
                    optimize=False,
                    kv_cache=kv_cache,
                )
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            shuffled_idx[:, [i]] = idx_next
            nn = reorder(shuffled_idx, o, reverse=True)[0].tolist()
            to_generate -= 1
            # decode and rewrite the prompt
            if should_print:
                refresh_lines(decode(nn), i, kv_cache.size)

            final = reorder(shuffled_idx, o, reverse=True)

        return final, num_tokens - prompt_len

    @torch.no_grad()
    def generate(
        self,
        idx,
        mask_idx,
        temperature=1.0,
        top_k=None,
        order_type=None,
        verbose=False,
        decode=None,
    ):
        """Generate tokens similarly than the original GPT but with a shuffling of the input sequence."""
        _, num_tokens = idx.size()
        o = order_mask(mask_idx, order_type=order_type)
        shuffled_idx = idx.gather(1, o).contiguous()

        should_print = verbose and decode is not None
        if should_print:
            init_prompt(num_tokens)

        to_generate = mask_idx.sum(1).max().item()
        prompt_len = num_tokens - to_generate

        for i in range(prompt_len, num_tokens):
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(shuffled_idx[:, :i], order=o, optimize=False)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            shuffled_idx[:, [i]] = idx_next
            nn = reorder(shuffled_idx, o, reverse=True)[0].tolist()
            to_generate -= 1
            # decode and rewrite the prompt
            if should_print:
                refresh_lines(decode(nn), i, shuffled_idx[:, :i].shape[1])

            final = reorder(shuffled_idx, o, reverse=True)

        return final, num_tokens - prompt_len


COLS = 190
MAX_BUFFER_SIZE = COLS * 20
LINES = (MAX_BUFFER_SIZE // COLS) + 1


def init_prompt(string):
    """Print a number of empty lines to clear the screen.

    Args:
        l (int): Number of lines to print.
    """
    if string < 1000:
        print("\n" * 4)

    else:
        print("\n" * LINES)


def process(s):
    """Process a string to make it printable.

    Remove all non-printable characters and replace newlines with a special character.

    Args:
        s (str): Input string.
    """
    s = s.replace("\n", "·")
    re.sub(r'[^a-zA-Z0-9 ,.!?;:\-\'"]', ",", s)

    return s


def split(s, length=64):
    """Split a string into lines of a given length."""
    return [s[i : i + length] for i in range(0, len(s), length)]


def refresh_lines(s, current=None, size=0, prompt=0):
    """Print to the terminal and refresh the screen.

    Args:
        s (str): String to print.
        current (int): Current number of accepts.
        size (int): Size of the kv_cache.
        prompt (int): Length of the prompt.
    """
    if len(s) > 450:
        s = process(s)
        s = s.replace("!", "  ·  ")
        seq = f"{s:-<{MAX_BUFFER_SIZE}}"[:MAX_BUFFER_SIZE]
        sys.stdout.write(f"\033[{LINES}A")

        ss = split(seq, COLS)

        if current is not None:
            print(
                f"Current: {current:<10}, len(s): {len(s):<10}, nlines: {len(ss):<10},"
                f" kv_cache.size: {size:<10} prompt: {prompt:<10}"
            )
        else:
            print(
                f"Lines: {LINES}, Columns: {COLS}, len(s): {len(s)}, nlines: {len(ss)}"
            )

        for line in ss:
            print(line)

        with open("output.txt", "a") as f:
            f.write('\n"')
            f.writelines(s)
            f.write('",\n')

    else:
        s = split(process(s))
        sys.stdout.write(f"\033[{len(s)+1}A")
        print(f"Total accepts: {current:<3}")
        for line in s:
            print(line)
