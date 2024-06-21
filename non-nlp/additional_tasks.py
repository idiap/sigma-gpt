# Copyright © <2024> Idiap Research Institute <contact@idiap.ch>

# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>

# SPDX-License-Identifier: AGPL-3.0-only
"""Additional tasks for the evaluation of the model.

It corresponds to an adapted version of the Maze, Vertical forecast
and the three toy tasks (Permutation, Step, Product) from the σ-GPT paper.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm

from generation_strategy import generate_strategies
from import_picoclvr import maze, tasks

generate_strategies = generate_strategies  # can be patched in main.py
PAD_TOKEN = 579
GENERATION_METHOD = ""


def padded_cropped_list(raw_list, length=500):
    """Pad or crop a list to a desired length.

    Args:
        l (list): list to pad or crop
        length (int, optional): desired length. Defaults to 500.

    Returns:
        list: padded or cropped list
    """
    r = [PAD_TOKEN] * length
    m = min(len(raw_list), length)
    r[:m] = raw_list[:m]
    assert len(r) == length
    return r


class Maze(tasks.Maze):
    """Maze task with a cache for the data."""

    def __init__(
        self,
        nb_train_samples,
        nb_test_samples,
        data_dir,
        batch_size,
        height,
        width,
        nb_walls,
        dist_min,
        device,
    ):
        """Override the __init__ method to use a cache for the data."""
        # no super().__init__ call, same but with a cache

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.device = device

        if data_dir is None:
            data_dir = "data"
            os.makedirs(data_dir, exist_ok=True)

        train_data_save_path = os.path.join(
            data_dir,
            f"train_data_S{nb_train_samples}_H{height}_W{width}_N{nb_walls}_D{dist_min}.pt",
        )

        if not os.path.exists(train_data_save_path):
            print("Creating train data")
            data = maze.create_maze_data(
                nb_train_samples,
                height=height,
                width=width,
                nb_walls=nb_walls,
                progress_bar=lambda x: tqdm.tqdm(
                    x, dynamic_ncols=True, desc="data-train"
                ),
            )
            torch.save(data, train_data_save_path)

        train_mazes, train_paths, _ = torch.load(train_data_save_path)
        self.train_input = self.map2seq(train_mazes.to(device), train_paths.to(device))

        test_data_save_path = os.path.join(
            data_dir,
            f"test_data_S{nb_test_samples}_H{height}_W{width}_N{nb_walls}_D{dist_min}.pt",
        )

        if not os.path.exists(test_data_save_path):
            print("Creating test data")
            data = maze.create_maze_data(
                nb_test_samples,
                height=height,
                width=width,
                nb_walls=nb_walls,
                progress_bar=lambda x: tqdm.tqdm(
                    x, dynamic_ncols=True, desc="data-test"
                ),
            )
            torch.save(data, test_data_save_path)

        test_mazes, test_paths, _ = torch.load(test_data_save_path)
        self.test_input = self.map2seq(test_mazes.to(device), test_paths.to(device))
        self.nb_codes = max(self.train_input.max(), self.test_input.max()) + 1

    @property
    def prompt_len(self):
        """Return the prompt length."""
        return self.height * self.width

    def compute_error(
        self,
        model,
        generate,
        split="train",
        nb_to_use=-1,
        deterministic_synthesis=False,
    ):
        """Compute the error of the model on the maze task.

        Args:
            model (torch.nn.Module): the model to evaluate
            generate (function): the generation function to use [left-to-right ar, shuffled ar, burst sampling]
            split (str, optional): the split to use. Defaults to "train".
            nb_to_use (int, optional): number of samples to use. Defaults to -1.
            deterministic_synthesis (bool, optional): whether to use deterministic synthesis. Defaults to False.

        Returns:
            tuple: number of total samples, number of correct samples, confusion matrix
        """
        nb_total, nb_correct = 0, 0
        count = torch.zeros(
            self.prompt_len,
            self.prompt_len,
            device=self.device,
            dtype=torch.int64,
        )

        for input in self.batches(split, nb_to_use):
            result = input.clone()
            ar_mask = result.new_zeros(result.size())
            ar_mask[:, self.height * self.width :] = 1
            result *= 1 - ar_mask
            generate(
                model,
                self.batch_size,
                result,
                ar_mask,
                deterministic_synthesis,
                progress_bar_desc=None,
                device=self.device,
            )
            mazes, paths = self.seq2map(result)
            path_correctness = maze.path_correctness(mazes, paths)
            nb_correct += path_correctness.long().sum()
            nb_total += mazes.size(0)

            optimal_path_lengths = (
                (input[:, self.height * self.width :] == maze.v_path).long().sum(1)
            )
            predicted_path_lengths = (
                (result[:, self.height * self.width :] == maze.v_path).long().sum(1)
            )
            optimal_path_lengths = optimal_path_lengths[path_correctness]
            predicted_path_lengths = predicted_path_lengths[path_correctness]
            count[optimal_path_lengths, predicted_path_lengths] += 1

        if count.max() == 0:
            count = None
        else:
            count = count[
                : count.sum(1).nonzero().max() + 1, : count.sum(0).nonzero().max() + 1
            ]

        return nb_total, nb_correct, count

    def produce_results(
        self, n_epoch, model, result_dir, logger, deterministic_synthesis
    ):
        """Produce results for the maze task for each split and generation strategy.

        Args:
            n_epoch (int): the current epoch
            model (torch.nn.Module): the model to evaluate
            result_dir (str): the directory to save the results
            logger (function): the logging function
            deterministic_synthesis (bool): whether to use deterministic synthesis
        """
        for gen_name, generate in generate_strategies.items():
            for split in ["train", "test"]:
                nb_total, nb_correct, count = self.compute_error(
                    model,
                    generate,
                    split,
                    nb_to_use=1000,
                    deterministic_synthesis=deterministic_synthesis,
                )
                logger(
                    f"accuracy_{gen_name}_{split} {n_epoch} nb_total {nb_total}"
                    f" nb_correct {nb_correct} accuracy {(100.0*nb_correct)/nb_total:.02f}%"
                )
                if count is not None:
                    proportion_optimal = count.diagonal().sum().float() / count.sum()
                    logger(
                        f"proportion_optimal_{gen_name}_{split} {n_epoch} {proportion_optimal*100:.02f}%"
                    )

            input = self.test_input[:48]
            result = input.clone()
            ar_mask = result.new_zeros(result.size())
            ar_mask[:, self.height * self.width :] = 1
            result *= 1 - ar_mask
            generate(
                model,
                self.batch_size,
                result,
                ar_mask,
                deterministic_synthesis,
                progress_bar_desc=None,
                device=self.device,
            )

            mazes, paths = self.seq2map(input)
            _, predicted_paths = self.seq2map(result)

            filename = os.path.join(result_dir, f"maze_{gen_name}_{n_epoch:04d}.png")
            maze.save_image(
                filename,
                mazes=mazes,
                target_paths=paths,
                predicted_paths=predicted_paths,
                path_correct=maze.path_correctness(mazes, predicted_paths),
                path_optimal=maze.path_optimality(paths, predicted_paths),
            )
            logger(f"wrote {filename}")


def VerticalDataset(path="data/vertical.csv"):
    """Load the vertical dataset from a CSV file.

    Tokens range:
    0 - 449 altitude tokens
    450 - 578 aircraft type tokens
    579 PAD token
    sequence: [1 aircraft token, 500 cfls, 500 alts]

    Args:
        path (str, optional): path to the CSV file. Defaults to "data/vertical.csv".

    Returns:
        tuple: train and test datasets
    """
    df = pd.read_csv(path)
    gp = df.groupby("FLIGHT_ID")

    actype = gp.AIRCRAFT_TYPE.first()
    actype_set = list(set(actype))  # 129 different aircraft types
    actype_tokens = [
        actype_set.index(a) + 450 for a in actype
    ]  # 450 different altitudes

    dataset = [
        [ac] + padded_cropped_list(cfl) + padded_cropped_list(alt)
        for ac, cfl, alt in zip(
            actype_tokens,
            list(gp.CFL.apply(list)),
            list(gp.TRACK_ALTITUDE.apply(list)),
        )
    ]

    return torch.tensor(dataset[:-1000]), torch.tensor(dataset[-1000:])


class Vertical(tasks.Task):
    """Predict the altitude of an aircraft given its cleared flight level (CFL) and aircraft type."""

    def __init__(self, batch_size, device, tqdm=True) -> None:
        """Initialize the task.

        Args:
            batch_size (int): batch size
            device (torch.device, optional): device to use. Defaults to torch.device("cpu").
            tqdm (bool, optional): whether to use tqdm. Defaults to True.
        """
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.prompt_len = 501 + 1
        self.tqdm = tqdm

        self.train_input, self.test_input = VerticalDataset()
        self.train_input = self.train_input.to(self.device)
        self.test_input = self.test_input.to(self.device)

    def vocabulary_size(self):
        """Return the vocabulary size."""
        return 580

    def batches(self, split="train", nb_to_use=-1, desc=None):
        """Generate batches of data.

        Args:
            split (str, optional): split to use. Defaults to "train".
            nb_to_use (int, optional): number of samples to use. Defaults to -1.
            desc (str, optional): description for the progress bar. Defaults to None.

        Yields:
            torch.Tensor: batch of data
        """
        assert split in {"train", "test"}
        input = self.train_input if split == "train" else self.test_input
        if nb_to_use > 0:
            input = input[:nb_to_use]
        if desc is None:
            desc = f"epoch-{split}"

        pbar = input.split(self.batch_size)

        if self.tqdm:
            pbar = tqdm.tqdm(pbar, dynamic_ncols=True, desc=desc)

        for batch in pbar:
            yield batch

    def compute_error(
        self,
        model,
        generate,
        split="train",
        nb_to_use=-1,
        deterministic_synthesis=False,
    ):
        """Compute the error of the model on the vertical task.

        Args:
            model (torch.nn.Module): the model to evaluate
            generate (function): the generation function to use [left-to-right ar, shuffled ar, burst sampling]
            split (str, optional): the split to use. Defaults to "train".
            nb_to_use (int, optional): number of samples to use. Defaults to -1.
            deterministic_synthesis (bool, optional): whether to use deterministic synthesis. Defaults to False.

        Returns:
            tuple: number of total samples, number of correct samples,
                    sum of mean squared errors, number of continuity error
        """
        nb_total, nb_correct, sum_mse, nb_continuous = 0, 0, 0.0, 0
        for input in self.batches(split, nb_to_use):
            result = input.clone()
            ar_mask = result.new_zeros(result.size())
            ar_mask[:, self.prompt_len :] = 1
            result *= 1 - ar_mask
            generate(
                model,
                self.batch_size,
                result,
                ar_mask,
                deterministic_synthesis,
                progress_bar_desc=None,
                device=self.device,
            )
            continuity, mse = self._correct_parts(input, result)
            corrects = continuity & mse
            sum_mse += mse.sum().item()
            nb_continuous += continuity.sum().item()

            nb_correct += corrects.sum().item()
            nb_total += input.size(0)

        return nb_total, nb_correct, sum_mse, nb_continuous

    def _correct_parts(
        self,
        input,
        prediction,
    ):
        """Compute the continuity and mean squared error of the prediction.

        Args:
            input (torch.Tensor): input data
            prediction (torch.Tensor): predicted data

        Returns:
            tuple: continuity, mean squared error
        """
        ref = input[:, self.prompt_len :].float()
        pred = prediction[:, self.prompt_len :].float()
        pad_mask = ref.eq(PAD_TOKEN)

        continuity = (pred[:, 1:] - pred[:, :-1]).abs()
        continuity[pad_mask[:, 1:]] = 0
        continuity = continuity.ge(10).any(1)
        mse = F.mse_loss(pred, ref, reduction="none")
        mse[pad_mask] = 0
        mse = mse.mean(1).ge(50)

        return continuity, mse

    def compute_error_and_log(
        self, n_epoch, model, split, gen_name, generate, logger, deterministic_synthesis
    ):
        """Compute the error of the model on the vertical task and log the results.

        Args:
            n_epoch (int): the current epoch
            model (torch.nn.Module): the model to evaluate
            split (str): the split to use
            gen_name (str): the generation strategy name
            generate (function): the generation function to use [left-to-right ar, shuffled ar, burst sampling]
            logger (function): the logging function
            deterministic_synthesis (bool): whether to use deterministic synthesis
        """
        nb_total, nb_correct, sum_mse, nb_continuous = self.compute_error(
            model,
            generate,
            split,
            nb_to_use=1000,
            deterministic_synthesis=deterministic_synthesis,
        )
        logger(
            f"accuracy_{gen_name}_{split} {n_epoch} nb_total {nb_total}"
            f" nb_correct {nb_correct} accuracy {(100.0*nb_correct)/nb_total:.02f}%"
        )
        logger(
            f"mse_{gen_name}_{split} {n_epoch} nb_total {nb_total} "
            f" sum_mse {sum_mse} mean_mse {sum_mse/nb_total:.02f}"
        )
        logger(
            f"continuity_{gen_name}_{split} {n_epoch} nb_total {nb_total}"
            f" nb_continuous {nb_continuous} continuity {(100.0*nb_continuous)/nb_total:.02f}%"
        )

    def produce_results(
        self, n_epoch, model, result_dir, logger, deterministic_synthesis
    ):
        """Produce results for the vertical task for each split and generation strategy.

        Args:
            n_epoch (int): the current epoch
            model (torch.nn.Module): the model to evaluate
            result_dir (str): the directory to save the results
            logger (function): the logging function
            deterministic_synthesis (bool): whether to use deterministic synthesis
        """
        for gen_name, generate in generate_strategies.items():
            for split in ["train", "test"]:
                self.compute_error_and_log(
                    n_epoch,
                    model,
                    split,
                    gen_name,
                    generate,
                    logger,
                    deterministic_synthesis,
                )

            input = self.test_input[:48]
            result = input.clone()
            ar_mask = result.new_zeros(result.size())
            ar_mask[:, self.prompt_len :] = 1
            result *= 1 - ar_mask
            generate(
                model,
                self.batch_size,
                result,
                ar_mask,
                deterministic_synthesis,
                progress_bar_desc=None,
                device=self.device,
            )
            continuity, mse = self._correct_parts(input, result)

            cls_name = self.__class__.__name__.lower()

            filename = os.path.join(
                result_dir, f"{cls_name}_{gen_name}_{n_epoch:04d}.pdf"
            )
            self.save_image(
                filename, input.cpu(), result.cpu(), (continuity + 2 * mse).long().cpu()
            )
            logger(f"wrote {filename}")

    def cfl(self, seq):
        """Return the cleared flight level part of the sequence [1:501].

        Args:
            seq (torch.Tensor): 1D sequence

        Returns:
            torch.Tensor: cleared flight level part of the sequence
        """
        return seq[1:501]

    def alt(self, seq):
        """Return the altitude part of the sequence [501:].

        Args:
            seq (torch.Tensor): 1D sequence

        Returns:
            torch.Tensor: altitude part of the sequence
        """
        return seq[501:]

    def save_image(self, filename, input, result, categs):
        """Save an image of the input and result sequences.

        color code:
            black: correct generation,
            lightgray: continuity error,
            purple: mse error,
            red: both errors

        Args:
            filename (str): path to the image file
            input (torch.Tensor): input sequence
            result (torch.Tensor): predicted sequence
            categs (torch.Tensor): categories
        """
        fig, axs = plt.subplots(8, 6, figsize=(8, 6), sharex=True, sharey=True)
        axs = axs.flatten()
        colors = ["black", "lightgray", "purple", "red"]

        for ax, i, r, c in zip(axs, input, result, categs):
            cfl = self.cfl(i)
            mask = cfl.ne(PAD_TOKEN)

            ref = self.alt(i)[mask]
            alt = self.alt(r)[mask]

            ax.plot(cfl, color="black", lw=0.5)
            ax.plot(alt, color="red")
            ax.plot(ref, color="black", linestyle=":", lw=0.5)

            ax.spines[["right", "top"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color(colors[c.item()])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylim(0, 500)

        fig.tight_layout()
        plt.savefig(filename)


def PermutationDataset(N: int, perm_length: int = 500):
    """Generate a dataset of permutations.

    Each sample is a permutation of the integers from 0 to perm_length - 1.

    Args:
        N (int): number of samples
        perm_length (int, optional): length of the permutation. Defaults to 500.
    """
    return torch.stack([torch.randperm(perm_length) for _ in range(N)])


def StepDataset(N: int, L: int = 500, length_step: int = 50):
    """Generate a dataset of steps.

    Each sample is a step of length length_step at a random position in a sequence of length L.

    Args:
        N (int): number of samples
        L (int, optional): length of the sequence. Defaults to 500.
        length_step (int, optional): length of the step. Defaults to 50.
    """
    idxs = torch.randint(0, L - length_step, (N,))[:, None] + torch.arange(length_step)
    idxs = idxs.clamp(0, L - 1)
    data = torch.zeros(N, L, dtype=torch.long).scatter_(1, idxs, 1)
    return data


def ProductDataset(N: int, L: int = 500, bernoulli_proba: float = 0.1):
    """Generate a dataset of products.

    Each sample is a sequence of length L with a product of bernoulli samples with probability bernoulli_proba.

    Args:
        N (int): number of samples
        L (int, optional): length of the sequence. Defaults to 500.
        bernoulli_proba (float, optional): probability of the bernoulli samples. Defaults to 0.1.
    """
    return torch.full((N, L), bernoulli_proba).bernoulli().long()


class ToyTask(tasks.Task):
    """Base class for toy tasks."""

    def batches(self, split="train", nb_to_use=-1, desc=None):
        """Generate batches of data.

        Args:
            split (str, optional): split to use. Defaults to "train".
            nb_to_use (int, optional): number of samples to use. Defaults to -1.
            desc (str, optional): description for the progress bar. Defaults to None.

        Yields:
            torch.Tensor: batch of data
        """
        assert split in {"train", "test"}
        input = self.train_input if split == "train" else self.test_input
        if nb_to_use > 0:
            input = input[:nb_to_use]
        if desc is None:
            desc = f"epoch-{split}"
        for batch in tqdm.tqdm(
            input.split(self.batch_size), dynamic_ncols=True, desc=desc
        ):
            yield batch

    def _correct(self, input, prediction):
        pass

    def save_image(results):
        """Save an image of the results.

        Args:
            results (torch.Tensor): results
        """
        pass

    def compute_error(
        self,
        model,
        generate,
        split="train",
        nb_to_use=-1,
        deterministic_synthesis=False,
    ):
        """Compute the error of the model on the toy task.

        Args:
            model (torch.nn.Module): the model to evaluate
            generate (function): the generation function to use [left-to-right ar, shuffled ar, burst sampling]
            split (str, optional): the split to use. Defaults to "train".
            nb_to_use (int, optional): number of samples to use. Defaults to -1.
            deterministic_synthesis (bool, optional): whether to use deterministic synthesis. Defaults to False.

        Returns:
            tuple: number of total samples, number of correct samples
        """
        nb_total, nb_correct = 0, 0
        for input in self.batches(split, nb_to_use):
            result = input.clone()
            ar_mask = result.new_ones(result.size())
            result *= 1 - ar_mask
            generate(
                model,
                self.batch_size,
                result,
                ar_mask,
                deterministic_synthesis,
                progress_bar_desc=None,
                device=self.device,
            )
            nb_correct += self._correct(input, result).sum().item()
            nb_total += input.size(0)

        return nb_total, nb_correct

    @torch.no_grad()
    def produce_results(
        self, n_epoch, model, result_dir, logger, deterministic_synthesis
    ):
        """Produce results for the toy task for each split and generation strategy.

        Args:
            n_epoch (int): the current epoch
            model (torch.nn.Module): the model to evaluate
            result_dir (str): the directory to save the results
            logger (function): the logging function
            deterministic_synthesis (bool): whether to use deterministic synthesis
        """
        for gen_name, generate in generate_strategies.items():
            for split in ["train", "test"]:
                nb_total, nb_correct = self.compute_error(
                    model,
                    generate,
                    split,
                    nb_to_use=1000,
                    deterministic_synthesis=deterministic_synthesis,
                )
                logger(
                    f"accuracy_{gen_name}_{split} {n_epoch} nb_total {nb_total}"
                    f" nb_correct {nb_correct} accuracy {(100.0*nb_correct)/nb_total:.02f}%"
                )

            input = self.test_input[:48]
            result = input.clone()
            ar_mask = result.new_ones(result.size())
            result *= 1 - ar_mask
            generate(
                model,
                self.batch_size,
                result,
                ar_mask,
                deterministic_synthesis,
                progress_bar_desc=None,
                device=self.device,
            )
            corrects = self._correct(input, result)

            cls_name = self.__class__.__name__.lower()

            filename = os.path.join(
                result_dir, f"{cls_name}_{gen_name}_{n_epoch:04d}.pdf"
            )
            self.save_image(filename, result.cpu(), corrects.cpu())
            logger(f"wrote {filename}")


class Permutation(ToyTask):
    """Predict the permutation of a sequence of integers."""

    def __init__(
        self,
        nb_train_samples,
        nb_test_samples,
        batch_size,
        seq_length,
        device,
        tqdm=True,
    ) -> None:
        """Initialize the task.

        Args:
            nb_train_samples (int): number of training samples
            nb_test_samples (int): number of test samples
            batch_size (int): batch size
            seq_length (int, optional): length of the sequence. Defaults to 100.
            device (torch.device, optional): device to use. Defaults to torch.device("cpu").
            tqdm (bool, optional): whether to use tqdm. Defaults to True.
        """
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.tqdm = tqdm

        self.train_input = PermutationDataset(nb_train_samples, seq_length)
        self.test_input = PermutationDataset(nb_test_samples, seq_length)

        self.train_input = self.train_input.to(self.device)
        self.test_input = self.test_input.to(self.device)

    def vocabulary_size(self):
        """Return the vocabulary size."""
        return self.seq_length

    def _correct(self, input, predicted_perm):
        """Check if the prediction is correct.

        A prediction is correct if it is a permutation of the input, with all elements in the input.

        Args:
            input (torch.Tensor): input data
            predicted_perm (torch.Tensor): predicted data

        Returns:
            torch.Tensor: tensor of booleans
        """
        return predicted_perm.sort()[0].eq(input.sort()[0]).all(dim=1)

    def save_image(self, filename, result, corrects):
        """Save an image of the results.

        The first image shows the counts of each element in the results, to check if there is a mode collapse.
        The second image shows the 3 first results sorted to check if all elements are present.
        The third image shows the number of different elements in the results.

        Args:
            filename (str): path to the image file
            result (torch.Tensor): results
            corrects (torch.Tensor): correctness
        """
        fig, axs = plt.subplots(1, 3, figsize=(15, 10), sharey=True)

        hots = F.one_hot(result, num_classes=100)
        counts = hots.sum(dim=0)
        axs[0].set_title("Counts")
        axs[0].imshow(counts, cmap="binary", origin="lower")

        # five first
        axs[1].set_title("First")
        axs[1].imshow(
            F.one_hot(result[0].sort().values, num_classes=100).T,
            cmap="binary",
            origin="lower",
            alpha=0.5,
        )
        axs[1].imshow(
            F.one_hot(result[1].sort().values, num_classes=100).T,
            cmap="binary",
            origin="lower",
            alpha=0.5,
        )
        axs[1].imshow(
            F.one_hot(result[2].sort().values, num_classes=100).T,
            cmap="binary",
            origin="lower",
            alpha=0.5,
        )

        # sorted count
        s = result.sort().values
        v = (100 - (s[:, 1:] == s[:, :-1]).sum(1)).sort(descending=True).values
        axs[2].set_title("Different elements")
        axs[2].bar([0], v[0], color="red")
        axs[2].bar(range(1, 48), v[1:], color="black")
        axs[2].axis("square")

        fig.tight_layout()

        plt.savefig(filename)


class Step(ToyTask):
    """Predict a step in a sequence."""

    def __init__(
        self,
        nb_train_samples,
        nb_test_samples,
        batch_size,
        seq_length,
        step_length,
        device,
        tqdm=True,
    ) -> None:
        """Initialize the task.

        Args:
            nb_train_samples (int): number of training samples
            nb_test_samples (int): number of test samples
            batch_size (int): batch size
            seq_length (int, optional): length of the sequence. Defaults to 100.
            step_length (int, optional): length of the step. Defaults to 100.
            device (torch.device, optional): device to use. Defaults to torch.device("cpu").
            tqdm (bool, optional): whether to use tqdm. Defaults to True.
        """
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.step_length = step_length
        self.tqdm = tqdm

        self.train_input = StepDataset(nb_train_samples, seq_length, step_length)
        self.test_input = StepDataset(nb_test_samples, seq_length, step_length)

        self.train_input = self.train_input.to(self.device)
        self.test_input = self.test_input.to(self.device)

    def vocabulary_size(self):
        """Return the vocabulary size (two classes 0-1)."""
        return 2

    def _correct(self, input, predicted_step):
        """Check if the prediction is correct.

        Step must be of correct length (consecutive 1s) and the rest 0s

        Args:
            input (torch.Tensor): input data
            predicted_step (torch.Tensor): predicted data

        Returns:
            torch.Tensor: tensor of booleans
        """
        diff = predicted_step[:, 1:] - predicted_step[:, :-1]
        steps = diff.abs().sum(dim=1).le(2)
        length = predicted_step.sum(dim=1).eq(self.step_length)
        return steps & length

    def save_image(self, filename, result, corrects):
        """Save an image of generations, correct in black, incorrect in red.

        Args:
            filename (str): path to the image file
            result (torch.Tensor): results
            corrects (torch.Tensor): correctness
        """
        fig, axs = plt.subplots(8, 6, figsize=(8, 6), sharex=True, sharey=True)
        axs = axs.flatten()

        for ax, d, c in zip(axs, result, corrects):
            ax.plot(d, color="black" if c else "red")
            ax.axis("off")

        fig.tight_layout()
        plt.savefig(filename)


class Product(ToyTask):
    """Predict a product law, sequence of Bernoulli samples."""

    def __init__(
        self,
        nb_train_samples,
        nb_test_samples,
        batch_size,
        seq_length,
        bernoulli_proba,
        device,
        tqdm=True,
    ) -> None:
        """Initialize the task.

        Args:
            nb_train_samples (int): number of training samples
            nb_test_samples (int): number of test samples
            batch_size (int): batch size
            seq_length (int, optional): length of the sequence. Defaults to 100.
            bernoulli_proba (float, optional): probability of the Bernoulli samples. Defaults to 0.1.
            device (torch.device, optional): device to use. Defaults to torch.device("cpu").
            tqdm (bool, optional): whether to use tqdm. Defaults to True.
        """
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.bernoulli_proba = bernoulli_proba
        self.tqdm = tqdm

        self.train_input = ProductDataset(nb_train_samples, seq_length, bernoulli_proba)
        self.test_input = ProductDataset(nb_test_samples, seq_length, bernoulli_proba)

        self.train_input = self.train_input.to(self.device)
        self.test_input = self.test_input.to(self.device)

    def vocabulary_size(self):
        """Return the vocabulary size (two classes 0-1)."""
        return 2

    def _correct(self, input, prediction):
        return wald_wolowitz_test(prediction, self.bernoulli_proba)

    def save_image(self, filename, result, corrects):
        """Save an image of generations, correct in black, incorrect in red.

        Last two plots show the distribution and the mean of the results to check for mode collapse.

        Args:
            filename (str): path to the image file
            result (torch.Tensor): results
            corrects (torch.Tensor): correctness
        """
        fig, axs = plt.subplots(9, 6, figsize=(9, 6), sharex=True, sharey=True)
        faxs = axs.flatten()

        for ax, d, c in zip(faxs, result, corrects):
            ax.plot(d, ",", color="black" if c else "red")
            ax.axis("off")

        gs = axs[-1, 0].get_gridspec()
        for ax in axs[-1, :]:
            ax.remove()

        bigs = [fig.add_subplot(gs[-1, :3]), fig.add_subplot(gs[-1, 3:])]
        for ax in bigs:
            ax.axis("off")

        m = result.float().mean()
        bigs[0].bar([0, 1], [1 - m, m], color="black", width=1.0)
        bigs[0].hlines(0.1, -0.5, 1.5, color="red")

        bigs[1].bar(range(self.seq_length), result.sum(0) / 48, color="black")

        fig.tight_layout()

        plt.savefig(filename)


def wald_wolowitz_test(result, bernoulli_p=0.5):
    """Wald-Wolowitz test to check if the sequence is likely to be a Bernoulli sequence.

    https://en.wikipedia.org/wiki/Wald%E2%80%93Wolfowitz_runs_test

    Args:
        result (torch.Tensor): results
        bernoulli_p (float, optional): Bernoulli probability. Defaults to 0.5.

    Returns:
        torch.Tensor: tensor of booleans
    """
    diff = result[:, 1:] - result[:, :-1]
    nb_runs = diff.abs().sum(1) + 1
    N = result.shape[1]
    n1 = bernoulli_p * N
    # n1 = seq.sum(1)
    n0 = N - n1
    mu = 2 * n1 * n0 / N + 1
    sigma = 2 * n1 * n0 * (2 * n1 * n0 - N) / (N**2 * (N - 1))

    z_score = (nb_runs - mu) / sigma

    p_value = 2 * (1 - torch.distributions.Normal(0, 1).cdf(z_score.abs()))

    return p_value > 0.001
