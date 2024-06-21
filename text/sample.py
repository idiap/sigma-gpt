# Copyright © 2022 Andrej Karpathy
# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>
# SPDX-FileContributor: Andrej Karpathy and nanogpt contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""
Adapted from nanoGPT sample.py for σ-GPT.

Sample from a trained model
"""

import os
import pickle
from contextlib import nullcontext

import tiktoken
import torch

from import_nanogpt import model
from sigmagpt import sigmaGPT

# change dir to the dir of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# If needed set HF CACHE
# os.environ["HF_DATASETS_CACHE"] = ...
# # os.environ["TRANSFORMERS_CACHE"] = ...

order_type = "random"  # 'left-to-right', 'fractal', 'random'
# -----------------------------------------------------------------------------
# either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
init_from = "resume"
out_dir = "out"  # ignored if init_from is not 'resume'
prompt = "\n"  # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10  # number of samples to draw
max_tokens = 1024  # number of tokens generated in each sample
# 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
temperature = 1.0
# retain only the top_k most likely tokens, clamp others to have 0 probability
# top_k = 200
top_k = None
seed = 1337
device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)  # 'float32' or 'bfloat16' or 'float16'
compile = False  # use PyTorch 2.0 to compile the model to be faster
verbose = True
use_kv = True
use_rejection_sampling = True

exec(
    open("nanoGPT/configurator.py").read()
)  # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# model
if init_from == "resume":
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    print(f"Loaded checkpoint from {ckpt_path}")
    print(f"iter_num = {checkpoint['iter_num']}")
    print(f"best_val_loss = {checkpoint['best_val_loss']}")
    gptconf = model.GPTConfig(**checkpoint["model_args"])
    model = sigmaGPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)  # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if (
    init_from == "resume"
    and "config" in checkpoint
    and "dataset" in checkpoint["config"]
):  # older checkpoints might not have these...
    meta_path = os.path.join(
        "nanoGPT/data", checkpoint["config"]["dataset"], "meta.pkl"
    )
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    stoi, itos = meta["stoi"], meta["itos"]

    def encode(string):
        """Encode a string using the vocabulary."""
        return [stoi.get(c, 0) for c in string]

    def decode(list_of_tokens):
        """Decode a list of integers using the vocabulary."""
        return "".join([itos[i] for i in list_of_tokens])

    meta_vocab_size = meta["vocab_size"]
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")

    def encode(string):
        """Encode a string using the vocabulary."""
        return enc.encode(string)

    def decode(list_of_tokens):
        """Decode a list of integers using the vocabulary."""
        return enc.decode(list_of_tokens)


# encode the beginning of the prompt
if prompt.startswith("FILE:"):
    with open(prompt[5:], "r", encoding="utf-8") as f:
        prompt = f.read()

prompt = prompt.replace("</missing>", "<missing>")

x = []
m = []
for seq in prompt.split("<missing>"):
    try:
        x.append(torch.zeros((1, int(seq)), dtype=torch.long, device=device))
        m.append(torch.ones((1, int(seq)), dtype=torch.long, device=device))
    except ValueError:
        ids = encode(seq)
        x.append(torch.tensor(ids, dtype=torch.long, device=device)[None, ...])
        m.append(torch.zeros((1, x[-1].shape[1]), dtype=torch.long, device=device))
x = torch.cat(x, dim=1)
m = torch.cat(m, dim=1)
x = x.repeat((1, 1))
m = m.repeat((1, 1))

full_x = torch.full(
    (x.size(0), max_tokens), fill_value=0, dtype=torch.long, device=device
)
full_x[:, : x.size(1)] = x
mask = torch.full(
    (x.size(0), max_tokens), fill_value=1, dtype=torch.long, device=device
)
mask[:, : x.size(1)] = m

print("\n" * 2)
if use_rejection_sampling:
    print("Generating with rejection sampling")
    gen = model.generate_rejection_sampling
else:
    if use_kv:
        print("Generating autoregressively with kv cache")
        gen = model.generate_autoregressively_with_kvcache
    else:
        print("Generating autoregressively without kv cache")
        gen = model.generate


# run generation
with torch.no_grad(), ctx:
    for _ in range(num_samples):
        y, k = gen(
            full_x.clone(),
            mask.clone(),
            temperature=temperature,
            top_k=top_k,
            order_type=order_type,
            verbose=verbose,
            decode=decode,
        )

        if not verbose:
            for yy in y:
                out = decode(yy.tolist())
                print(out)

        print(f"---------------:{k}steps.")
