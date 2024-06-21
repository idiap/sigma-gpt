# Copyright © Public Domain
# SPDX-FileContributor: Francois Fleuret <francois@fleuret.org>
# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>
# SPDX-License-Identifier: CC0-1.0
"""Adapted version of the Picoclvr project for the σ-GPT paper."""
import argparse
import datetime
import math
import os
import sys
import time

import torch
from torch.nn import functional as F

import additional_tasks
from generation_strategy import generate_strategies
from import_picoclvr import problems, tasks
from sigmagpt import SigmaGPT

######################################################################

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
else:
    device = torch.device("cpu")

######################################################################

parser = argparse.ArgumentParser(
    description="An implementation of GPT with cache.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)


parser.add_argument(
    "--task",
    type=str,
    default="twotargets",
    help="all the awesome tasks from initial picoclvr project (including maze)"
    "+ product, permutation, step, vertical",
)


######################################
# Additional options

parser.add_argument(
    "--training_strategy",
    choices=["left-to-right", "shuffle", "fractal"],
    default=None,
    required=True,
)

parser.add_argument(
    "--generation_strategy",
    choices=["left-to-right", "shuffle", "fractal", "burst"],
    nargs="*",
)

parser.add_argument("--maze_data_dir", type=str, default=None)

parser.add_argument("--maze_dist_min", type=int, default=10)

######################################


parser.add_argument("--synt_sequence_length", type=int, default=100)

parser.add_argument("--bernoulli_proba", type=float, default=0.1)

parser.add_argument("--step_length", type=int, default=10)


parser.add_argument("--log_filename", type=str, default="train.log", help=" ")

parser.add_argument("--result_dir", type=str, default=None)

parser.add_argument("--seed", type=int, default=0)

parser.add_argument("--max_percents_of_test_in_train", type=int, default=1)

########################################

parser.add_argument("--nb_epochs", type=int, default=25)

parser.add_argument("--batch_size", type=int, default=None)

parser.add_argument("--nb_train_samples", type=int, default=None)

parser.add_argument("--nb_test_samples", type=int, default=None)

parser.add_argument("--optim", type=str, default="adam")

parser.add_argument("--learning_rate", type=float, default=1e-4)

parser.add_argument("--learning_rate_schedule", type=str, default="10: 2e-5,30: 4e-6")

########################################

parser.add_argument("--model", type=str, default=None)

parser.add_argument("--dim_model", type=int, default=None)

parser.add_argument("--dim_keys", type=int, default=None)

parser.add_argument("--dim_hidden", type=int, default=None)

parser.add_argument("--nb_heads", type=int, default=None)

parser.add_argument("--nb_blocks", type=int, default=None)

parser.add_argument("--dropout", type=float, default=0.1)

########################################

parser.add_argument("--deterministic_synthesis", action="store_true", default=False)

parser.add_argument("--no_checkpoint", action="store_true", default=False)

parser.add_argument("--overwrite_results", action="store_true", default=False)

parser.add_argument("--checkpoint_name", type=str, default="checkpoint.pth")

parser.add_argument("--tqdm", action=argparse.BooleanOptionalAction, default=True)


##############################
# Original picoclvr options

##############################
# filetask

parser.add_argument("--filetask_train_file", type=str, default=None)

parser.add_argument("--filetask_test_file", type=str, default=None)

##############################
# rpl options

parser.add_argument("--rpl_nb_starting_values", type=int, default=3)

parser.add_argument("--rpl_max_input", type=int, default=9)

parser.add_argument("--rpl_prog_len", type=int, default=8)

parser.add_argument("--rpl_nb_runs", type=int, default=5)

parser.add_argument("--rpl_no_prog", action="store_true", default=False)

##############################
# grid options

parser.add_argument("--grid_size", type=int, default=6)

parser.add_argument("--grid_fraction_play", type=float, default=0)

##############################
# picoclvr options

parser.add_argument("--picoclvr_nb_colors", type=int, default=5)

parser.add_argument("--picoclvr_height", type=int, default=12)

parser.add_argument("--picoclvr_width", type=int, default=16)

parser.add_argument("--picocvlr_prune_properties", type=str, default="none")

##############################
# Maze options

parser.add_argument("--maze_height", type=int, default=13)

parser.add_argument("--maze_width", type=int, default=21)

parser.add_argument("--maze_nb_walls", type=int, default=15)

##############################
# Snake options

parser.add_argument("--snake_height", type=int, default=9)

parser.add_argument("--snake_width", type=int, default=12)

parser.add_argument("--snake_nb_colors", type=int, default=5)

parser.add_argument("--snake_length", type=int, default=200)

##############################
# Stack options

parser.add_argument("--stack_nb_steps", type=int, default=100)

parser.add_argument("--stack_nb_stacks", type=int, default=3)

parser.add_argument("--stack_nb_digits", type=int, default=3)

parser.add_argument("--stack_fraction_values_for_train", type=float, default=0.75)

##############################
# Expr options

parser.add_argument("--expr_nb_variables", type=int, default=5)

parser.add_argument("--expr_sequence_length", type=int, default=40)

parser.add_argument("--expr_operand_max", type=int, default=9)

parser.add_argument("--expr_result_max", type=int, default=99)

parser.add_argument("--expr_input_file", type=str, default=None)

##############################
# Mixing

parser.add_argument("--mixing_hard", action="store_true", default=False)

parser.add_argument("--mixing_deterministic_start", action="store_true", default=False)

##############################
# greed options

parser.add_argument("--greed_height", type=int, default=5)

parser.add_argument("--greed_width", type=int, default=7)

parser.add_argument("--greed_T", type=int, default=25)

parser.add_argument("--greed_nb_walls", type=int, default=5)

parser.add_argument("--greed_nb_coins", type=int, default=2)

######################################################################

args = parser.parse_args()

# No need to generate in all the strategies if we are not training
# a sigma-GPT, unless explicitly asked for
if args.training_strategy != "shuffle":
    if args.generation_strategy is None:
        args.generation_strategy = [args.training_strategy]

# Set the list of generation strategies for the additional tasks
if args.generation_strategy is not None:
    additional_tasks.generate_strategies = {
        k: generate_strategies[k] for k in args.generation_strategy
    }


# For the old task, set generation to the training strategy
tasks.masked_inplace_autoregression = generate_strategies[args.training_strategy]


assert args.picocvlr_prune_properties in {"none", "train+eval", "eval"}

if args.result_dir is None:
    args.result_dir = f"results_{args.task}_{args.training_strategy}"

######################################################################

default_task_args = {
    "file": {
        "model": "37M",
        "batch_size": 25,
        "nb_train_samples": 250000,
        "nb_test_samples": 10000,
    },
    "addition": {
        "model": "352M",
        "batch_size": 25,
        "nb_train_samples": 250000,
        "nb_test_samples": 10000,
    },
    "byheart": {
        "model": "37M",
        "batch_size": 25,
        "nb_train_samples": 50000,
        "nb_test_samples": 10000,
    },
    "expr": {
        "model": "352M",
        "batch_size": 25,
        "nb_train_samples": 2500000,
        "nb_test_samples": 10000,
    },
    "grid": {
        "model": "37M",
        "batch_size": 25,
        "nb_train_samples": 250000,
        "nb_test_samples": 10000,
    },
    "qmlp": {
        "model": "37M",
        "batch_size": 10,
        "nb_train_samples": 100000,
        "nb_test_samples": 1000,
    },
    "guessop": {
        "model": "352M",
        "batch_size": 25,
        "nb_train_samples": 1000000,
        "nb_test_samples": 10000,
    },
    "learnop": {
        "model": "37M",
        "batch_size": 25,
        "nb_train_samples": 50000,
        "nb_test_samples": 10000,
    },
    "maze": {
        "model": "37M",
        "batch_size": 5,
        "nb_train_samples": 100000,
        "nb_test_samples": 10000,
    },
    "picoclvr": {
        "model": "37M",
        "batch_size": 25,
        "nb_train_samples": 250000,
        "nb_test_samples": 10000,
    },
    "rpl": {
        "model": "352M",
        "batch_size": 5,
        "nb_train_samples": 2500000,
        "nb_test_samples": 10000,
    },
    "snake": {
        "model": "37M",
        "batch_size": 25,
        "nb_train_samples": 250000,
        "nb_test_samples": 10000,
    },
    "stack": {
        "model": "37M",
        "batch_size": 25,
        "nb_train_samples": 100000,
        "nb_test_samples": 1000,
    },
    "twotargets": {
        "model": "37M",
        "batch_size": 25,
        "nb_train_samples": 50000,
        "nb_test_samples": 10000,
    },
    "memory": {
        "model": "37M",
        "batch_size": 100,
        "nb_train_samples": 25000,
        "nb_test_samples": 1000,
    },
    "mixing": {
        "model": "37M",
        "batch_size": 25,
        "nb_train_samples": 250000,
        "nb_test_samples": 10000,
    },
    "mnist": {
        "model": "37M",
        "batch_size": 10,
        "nb_train_samples": 60000,
        "nb_test_samples": 10000,
    },
    "greed": {
        "model": "37M",
        "batch_size": 25,
        "nb_train_samples": 25000,
        "nb_test_samples": 10000,
    },
    "product": {
        "model": "4M",
        "batch_size": 25,
        "nb_train_samples": 250000,
        "nb_test_samples": 10000,
    },
    "step": {
        "model": "4M",
        "batch_size": 25,
        "nb_train_samples": 250000,
        "nb_test_samples": 10000,
    },
    "permutation": {
        "model": "4M",
        "batch_size": 25,
        "nb_train_samples": 250000,
        "nb_test_samples": 10000,
    },
    "vertical": {
        "model": "4M",
        "batch_size": 25,
        "nb_train_samples": 250000,
        "nb_test_samples": 10000,
    },
}

if args.task in default_task_args:
    for k, v in default_task_args[args.task].items():
        if getattr(args, k) is None:
            setattr(args, k, v)

######################################################################

default_model_args = {
    "17K": {
        "dim_model": 32,
        "dim_keys": 32,
        "dim_hidden": 32,
        "nb_heads": 2,
        "nb_blocks": 2,
    },
    "4M": {
        "dim_model": 256,
        "dim_keys": 32,
        "dim_hidden": 1024,
        "nb_heads": 4,
        "nb_blocks": 6,
    },
    "37M": {
        "dim_model": 512,
        "dim_keys": 64,
        "dim_hidden": 2048,
        "nb_heads": 8,
        "nb_blocks": 12,
    },
    "122M": {
        "dim_model": 768,
        "dim_keys": 64,
        "dim_hidden": 2048,
        "nb_heads": 8,
        "nb_blocks": 24,
    },
    "352M": {
        "dim_model": 1024,
        "dim_keys": 64,
        "dim_hidden": 2048,
        "nb_heads": 8,
        "nb_blocks": 48,
    },
}

if args.model in default_model_args:
    for k, v in default_model_args[args.model].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
else:
    raise ValueError(f"Unknown model {args.model}")

######################################################################

try:
    os.mkdir(args.result_dir)
except FileExistsError:
    if not args.overwrite_results:
        print(
            f"result directory {args.result_dir} already exists, please add --overwrite_results for continuing."
        )
        exit(1)

log_file = open(os.path.join(args.result_dir, args.log_filename), "a")

if args.seed >= 0:
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

######################################################################


def log_string(s):
    """Log a string to the console and to a file."""
    t = time.strftime("%Y%m%d-%H:%M:%S ", time.localtime())

    if log_file is not None:
        log_file.write(t + s + "\n")
        log_file.flush()

    print(t + s)
    sys.stdout.flush()


log_string(f"argv {' '.join(sys.argv)}")

for n in vars(args):
    log_string(f"args.{n} {getattr(args, n)}")


######################################################################


def picoclvr_pruner_horizontal_green(p):
    """Prune the picoclvr task by removing the horizontal green properties."""
    return not ("green" in p and ("left" in p or "right" in p))


picoclvr_pruner_train = (
    picoclvr_pruner_horizontal_green
    if args.picocvlr_prune_properties in {"train+eval"}
    else None
)

picoclvr_pruner_eval = (
    (lambda p: not picoclvr_pruner_horizontal_green(p))
    if args.picocvlr_prune_properties in {"train+eval", "eval"}
    else None
)
######################################################################
if args.task == "vertical":
    task = additional_tasks.Vertical(
        batch_size=args.batch_size, device=device, tqdm=args.tqdm
    )

elif args.task == "product":
    task = additional_tasks.Product(
        nb_train_samples=args.nb_train_samples,
        nb_test_samples=args.nb_test_samples,
        seq_length=args.synt_sequence_length,
        batch_size=args.batch_size,
        device=device,
        bernoulli_proba=args.bernoulli_proba,
        tqdm=args.tqdm,
    )
    args.max_percents_of_test_in_train = -1


elif args.task == "step":
    task = additional_tasks.Step(
        nb_train_samples=args.nb_train_samples,
        nb_test_samples=args.nb_test_samples,
        seq_length=args.synt_sequence_length,
        step_length=args.step_length,
        batch_size=args.batch_size,
        device=device,
        tqdm=args.tqdm,
    )
    args.max_percents_of_test_in_train = -1

elif args.task == "permutation":
    task = additional_tasks.Permutation(
        nb_train_samples=args.nb_train_samples,
        nb_test_samples=args.nb_test_samples,
        seq_length=args.synt_sequence_length,
        batch_size=args.batch_size,
        device=device,
        tqdm=args.tqdm,
    )
    args.max_percents_of_test_in_train = -1

elif args.task == "maze":
    task = additional_tasks.Maze(
        nb_train_samples=args.nb_train_samples,
        nb_test_samples=args.nb_test_samples,
        batch_size=args.batch_size,
        data_dir=args.maze_data_dir,
        height=args.maze_height,
        width=args.maze_width,
        nb_walls=args.maze_nb_walls,
        dist_min=args.maze_dist_min,
        device=device,
    )


######################################################################

elif args.task == "file":
    assert (
        args.filetask_train_file is not None and args.filetask_test_file is not None
    ), "You have to specify the task train and test files"
    task = tasks.TaskFromFile(
        args.filetask_train_file,
        args.filetask_test_file,
        nb_train_samples=args.nb_train_samples,
        nb_test_samples=args.nb_test_samples,
        batch_size=args.batch_size,
        shuffle=True,
        device=device,
    )
    args.max_percents_of_test_in_train = 0

elif args.task == "byheart":
    task = tasks.SandBox(
        problem=problems.ProblemByHeart(),
        nb_train_samples=args.nb_train_samples,
        nb_test_samples=args.nb_test_samples,
        batch_size=args.batch_size,
        logger=log_string,
        device=device,
    )
    args.max_percents_of_test_in_train = -1

elif args.task == "learnop":
    task = tasks.SandBox(
        problem=problems.ProblemLearnOperator(),
        nb_train_samples=args.nb_train_samples,
        nb_test_samples=args.nb_test_samples,
        batch_size=args.batch_size,
        logger=log_string,
        device=device,
    )


elif args.task == "guessop":
    task = tasks.SandBox(
        problem=problems.ProblemGuessOperator(),
        nb_train_samples=args.nb_train_samples,
        nb_test_samples=args.nb_test_samples,
        batch_size=args.batch_size,
        logger=log_string,
        device=device,
    )


elif args.task == "twotargets":
    task = tasks.SandBox(
        problem=problems.ProblemTwoTargets(),
        nb_train_samples=args.nb_train_samples,
        nb_test_samples=args.nb_test_samples,
        batch_size=args.batch_size,
        logger=log_string,
        device=device,
    )

elif args.task == "memory":
    task = tasks.SandBox(
        problem=problems.ProblemMemory(),
        nb_train_samples=args.nb_train_samples,
        nb_test_samples=args.nb_test_samples,
        batch_size=args.batch_size,
        logger=log_string,
        device=device,
    )

elif args.task == "mixing":
    task = tasks.SandBox(
        problem=problems.ProblemMixing(
            hard=args.mixing_hard, random_start=not args.mixing_deterministic_start
        ),
        nb_train_samples=args.nb_train_samples,
        nb_test_samples=args.nb_test_samples,
        batch_size=args.batch_size,
        logger=log_string,
        device=device,
    )

elif args.task == "addition":
    task = tasks.SandBox(
        problem=problems.ProblemAddition(),
        nb_train_samples=args.nb_train_samples,
        nb_test_samples=args.nb_test_samples,
        batch_size=args.batch_size,
        logger=log_string,
        device=device,
    )

elif args.task == "picoclvr":
    task = tasks.PicoCLVR(
        nb_train_samples=args.nb_train_samples,
        nb_test_samples=args.nb_test_samples,
        batch_size=args.batch_size,
        height=args.picoclvr_height,
        width=args.picoclvr_width,
        nb_colors=args.picoclvr_nb_colors,
        logger=log_string,
        device=device,
        pruner_train=picoclvr_pruner_train,
        pruner_eval=picoclvr_pruner_eval,
    )

elif args.task == "mnist":
    task = tasks.MNIST(
        nb_train_samples=args.nb_train_samples,
        nb_test_samples=args.nb_test_samples,
        batch_size=args.batch_size,
        device=device,
    )

# elif args.task == "maze":
#     task = tasks.Maze(
#         nb_train_samples=args.nb_train_samples,
#         nb_test_samples=args.nb_test_samples,
#         batch_size=args.batch_size,
#         height=args.maze_height,
#         width=args.maze_width,
#         nb_walls=args.maze_nb_walls,
#         device=device,
#     )

elif args.task == "snake":
    task = tasks.Snake(
        nb_train_samples=args.nb_train_samples,
        nb_test_samples=args.nb_test_samples,
        batch_size=args.batch_size,
        height=args.snake_height,
        width=args.snake_width,
        nb_colors=args.snake_nb_colors,
        length=args.snake_length,
        prompt_length=args.snake_length // 2,
        device=device,
    )

elif args.task == "stack":
    task = tasks.Stack(
        nb_train_samples=args.nb_train_samples,
        nb_test_samples=args.nb_test_samples,
        batch_size=args.batch_size,
        logger=log_string,
        nb_steps=args.stack_nb_steps,
        nb_stacks=args.stack_nb_stacks,
        nb_digits=args.stack_nb_digits,
        fraction_values_for_train=args.stack_fraction_values_for_train,
        device=device,
    )

elif args.task == "expr":
    task = tasks.Expr(
        nb_train_samples=args.nb_train_samples,
        nb_test_samples=args.nb_test_samples,
        nb_variables=args.expr_nb_variables,
        sequence_length=args.expr_sequence_length,
        operand_max=args.expr_operand_max,
        result_max=args.expr_result_max,
        batch_size=args.batch_size,
        device=device,
    )

elif args.task == "rpl":
    task = tasks.RPL(
        nb_train_samples=args.nb_train_samples,
        nb_test_samples=args.nb_test_samples,
        batch_size=args.batch_size,
        nb_starting_values=args.rpl_nb_starting_values,
        max_input=args.rpl_max_input,
        prog_len=args.rpl_prog_len,
        nb_runs=args.rpl_nb_runs,
        no_prog=args.rpl_no_prog,
        logger=log_string,
        device=device,
    )

elif args.task == "grid":
    task = tasks.Grid(
        nb_train_samples=args.nb_train_samples,
        nb_test_samples=args.nb_test_samples,
        batch_size=args.batch_size,
        size=args.grid_size,
        fraction_play=args.grid_fraction_play,
        logger=log_string,
        device=device,
    )

elif args.task == "qmlp":
    task = tasks.QMLP(
        nb_train_samples=args.nb_train_samples,
        nb_test_samples=args.nb_test_samples,
        batch_size=args.batch_size,
        result_dir=args.result_dir,
        logger=log_string,
        device=device,
    )

elif args.task == "greed":
    task = tasks.Greed(
        nb_train_samples=args.nb_train_samples,
        nb_test_samples=args.nb_test_samples,
        batch_size=args.batch_size,
        height=args.greed_height,
        width=args.greed_width,
        T=args.greed_T,
        nb_walls=args.greed_nb_walls,
        nb_coins=args.greed_nb_coins,
        logger=log_string,
        device=device,
    )

else:
    raise ValueError(f"Unknown task {args.task}")

##############################

if args.learning_rate_schedule == "cos":
    learning_rate_schedule = {}
    for n_epoch in range(args.nb_epochs):
        u = n_epoch / args.nb_epochs * math.pi
        learning_rate_schedule[n_epoch] = args.learning_rate * 0.5 * (1 + math.cos(u))
else:
    u = {
        int(k): float(v)
        for k, v in [
            tuple(x.split(":")) for x in args.learning_rate_schedule.split(",")
        ]
    }

    learning_rate_schedule = {}
    learning_rate = args.learning_rate
    for n_epoch in range(args.nb_epochs):
        if n_epoch in u:
            learning_rate = u[n_epoch]
        learning_rate_schedule[n_epoch] = learning_rate

log_string(f"learning_rate_schedule {learning_rate_schedule}")


######################################################################

log_string(f"device {device}")

vocabulary_size = task.vocabulary_size()

log_string(f"vocabulary_size {vocabulary_size}")

##############################

model = SigmaGPT(
    vocabulary_size=vocabulary_size,
    dim_model=args.dim_model,
    dim_keys=args.dim_keys,
    dim_hidden=args.dim_hidden,
    nb_heads=args.nb_heads,
    nb_blocks=args.nb_blocks,
    causal=True,
    dropout=args.dropout,
)

model.to(device)

nb_parameters = sum(p.numel() for p in model.parameters())
log_string(f"nb_parameters {nb_parameters} ({int(nb_parameters/1e6)}M)")

if args.optim == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
elif args.optim == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
elif args.optim == "adamw":
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
else:
    raise ValueError(f"Unknown optimizer {args.optim}.")

######################################################################

nb_epochs_finished = 0

if args.no_checkpoint:
    log_string("not trying to load checkpoint.")

else:
    try:
        checkpoint_name = os.path.join(args.result_dir, args.checkpoint_name)
        checkpoint = torch.load(checkpoint_name)
        nb_epochs_finished = checkpoint["nb_epochs_finished"]
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])

        try:
            torch.set_rng_state(checkpoint["rng_state"])
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])
        except Exception as e:
            log_string(f"error when setting the rng state: {e}")

        log_string(f"checkpoint loaded with {nb_epochs_finished} epochs finished.")

    except FileNotFoundError:
        log_string("starting from scratch.")

    except Exception as e:
        log_string(f"error when loading the checkpoint: {e}")
        exit(1)

######################################################################

if args.task == "expr" and args.expr_input_file is not None:
    task.produce_results(
        n_epoch=nb_epochs_finished,
        model=model,
        result_dir=args.result_dir,
        logger=log_string,
        deterministic_synthesis=args.deterministic_synthesis,
        input_file=args.expr_input_file,
    )

    exit(0)

######################################################################

# Compute the entropy of the training tokens

token_count = 0
for input in task.batches(split="train", desc="train-entropy"):
    token_count += F.one_hot(input, num_classes=task.vocabulary_size()).sum((0, 1))
token_probas = token_count / token_count.sum()
entropy = -torch.xlogy(token_probas, token_probas).sum()
train_set_perplexity = math.exp(entropy)

######################################################################
# A bit of paranoia never hurts

if args.max_percents_of_test_in_train >= 0:

    def subsets_as_tuples(batches, cs):
        """Yield subsets of batches as tuples.

        This function is used to check which percentage of the test
        samples are in the train set.
        """
        s = set()
        for batch in batches:
            for x in batch:
                s.add(tuple([v.item() for v in x]))
                if len(s) == cs:
                    yield s
                    s = set()
        yield s

    nb_test, nb_in_train = 0, 0
    for test_subset in subsets_as_tuples(
        task.batches(split="test", desc="test-check"), 25000
    ):
        in_train = set()
        for train_subset in subsets_as_tuples(
            task.batches(split="train", desc="train-check"), 25000
        ):
            in_train.update(test_subset.intersection(train_subset))
        nb_in_train += len(in_train)
        nb_test += len(test_subset)

    log_string(
        f"data_check {nb_in_train*100/nb_test:.02f}% ({nb_in_train}/{nb_test}) of test samples are in the train set"
    )

    assert (
        nb_in_train <= args.max_percents_of_test_in_train * nb_test / 100
    ), f"More than {args.max_percents_of_test_in_train}% of test samples are in the train set"


nb_samples_seen = 0

if nb_epochs_finished >= args.nb_epochs:
    task.produce_results(
        n_epoch=nb_epochs_finished,
        model=model,
        result_dir=args.result_dir,
        logger=log_string,
        deterministic_synthesis=args.deterministic_synthesis,
    )

time_pred_result = None

prompt_len = getattr(task, "prompt_length", 0)

for n_epoch in range(nb_epochs_finished, args.nb_epochs):
    learning_rate = learning_rate_schedule[n_epoch]

    log_string(f"learning_rate {learning_rate}")

    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate

    model.train()

    nb_train_samples, acc_train_loss = 0, 0.0

    for input in task.batches(split="train"):
        input = input.to(device)
        loss = model.compute_loss(
            input, prompt_len=prompt_len, order_type=args.training_strategy
        )
        acc_train_loss += loss.item() * input.size(0)
        nb_train_samples += input.size(0)
        nb_samples_seen += input.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.autograd.no_grad():
        model.eval()

        nb_test_samples, acc_test_loss = 0, 0.0

        for input in task.batches(split="test"):
            input = input.to(device)
            loss = model.compute_loss(
                input, prompt_len=prompt_len, order_type=args.training_strategy
            )
            acc_test_loss += loss.item() * input.size(0)
            nb_test_samples += input.size(0)

        train_perplexity = math.exp(min(100, acc_train_loss / nb_train_samples))
        test_perplexity = math.exp(min(100, acc_test_loss / nb_test_samples))

        log_string(
            f"perplexity {n_epoch} train_set {train_set_perplexity}"
            f" train_prediction {train_perplexity} test_prediction {test_perplexity}"
        )

        task.produce_results(
            n_epoch=n_epoch,
            model=model,
            result_dir=args.result_dir,
            logger=log_string,
            deterministic_synthesis=args.deterministic_synthesis,
        )

        time_current_result = datetime.datetime.now()
        if time_pred_result is not None:
            log_string(
                f"next_result {time_current_result + (time_current_result - time_pred_result)}"
            )
        time_pred_result = time_current_result

    checkpoint = {
        "nb_epochs_finished": n_epoch + 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "rng_state": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state()

    checkpoint_name = os.path.join(args.result_dir, args.checkpoint_name)
    torch.save(checkpoint, checkpoint_name)
    log_string(f"saved checkpoint {checkpoint_name}")

######################################################################
