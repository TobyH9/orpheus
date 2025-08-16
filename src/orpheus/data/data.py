import torch
import numpy as np

import lightning as pl


class CollateFn:

    def __init__(self, char_to_int: dict, int_to_char: dict):
        self.char_to_int = char_to_int
        self.int_to_char = int_to_char

    def __call__(self, batch):
        # batch: List[Tuple[List[str], List[str]]]
        input_batch = torch.stack(
            [self.convert_to_integers(x) for x, _ in batch]
        )  # (B, block_size)
        output_batch = torch.stack(
            [self.convert_to_integers(y) for _, y in batch]
        )  # (B, block_size)

        return input_batch, output_batch

    def convert_to_integers(self, seq):
        # map each char to an int and make a 1D LongTensor
        return torch.tensor([self.char_to_int[char] for char in seq], dtype=torch.long)

    def convert_to_characters(self, seq):
        # map each int to a char and make a 1D LongTensor
        return torch.tensor([self.int_to_cha[integer] for integer in seq])


class TinyShakeDataset(torch.utils.data.Dataset):
    """Tiny Shakespeare Dataset Class."""

    def __init__(self, data_path: str, split: str, block_size: int):

        super().__init__()

        self.data_path = data_path
        self.split = split
        self.block_size = block_size
        # Load all data upfront for simplicity
        self.data, self.characters, self.vocab_size = self._load_data()

    def _load_data(self) -> tuple[str, list[str], int]:
        """
        Reads the data in the specified path and loads it, depending on the specified
        split, 'train' or 'val'. It also provides a string containing the characters
        in the file provided, as well as the number of different characters (vocab size).

        Args:
            None

        Returns:
            data (str): a string containing the fraction of text in the given file
            relating to the correct split, 'train' or 'val'.
            characters (list[str]): a sorted list of strings containing all the characters used in "text"
            vocab_size (int): the length of "characters".
        """
        with open(self.data_path, "r", encoding="utf-8") as f:
            text = f.read()
            split_idx = int(0.9 * len(text))
            training_data = text[:split_idx]
            val_data = text[split_idx:]
        characters = sorted(list(set(text)))  # convert the text to a sorted list
        vocab_size = len(characters)

        if self.split == "train":
            data = training_data
        elif self.split == "val":
            data = val_data
        else:
            raise ValueError(
                "split must be 'train' or 'val', invalid entry was provided."
            )

        return data, characters, vocab_size

    def __len__(self):
        # last starting index that permits a full block plus target shift
        return max(0, len(self.data) - self.block_size - 1)

    def __getitem__(self, idx: int):
        """
        Get a random block from the training or validation data along with
        the predicted output for this block.

        Args:
            split (str): "train" or "val"

        Returns:
            Tuple[List[str], List[str]]: input block characters and target block characters
        """

        x = np.array([self.data[idx + i] for i in range(self.block_size)])
        y = np.array([self.data[idx + 1 + i] for i in range(self.block_size)])
        return x, y


class TinyShakeDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for STRING protein interaction data."""

    def __init__(
        self,
        data_path: str,
        block_size: int,
        batch_size: int,
    ):
        super().__init__()

        self.data_path = data_path
        self.block_size = block_size
        self.batch_size = batch_size

        self.train_dataset = TinyShakeDataset(
            data_path=self.data_path,
            split="train",
            block_size=self.block_size,
        )

        self.val_dataset = TinyShakeDataset(
            data_path=self.data_path,
            split="val",
            block_size=self.block_size,
        )

        self.characters = sorted(
            set(self.train_dataset.characters + self.val_dataset.characters)
        )
        self.char_to_int = {
            ch: i for i, ch in enumerate(self.characters)
        }  # the mapping from the characters in the dataset to integers
        self.int_to_char = {
            i: ch for i, ch in enumerate(self.characters)
        }  # the mapping from the integers to the characters in the dataset

    def setup(self, stage: str = None):
        """Set up datasets for training and validation."""
        pass

    def train_dataloader(self):
        """Create training data loader."""
        assert self.train_dataset is not None

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=CollateFn(
                char_to_int=self.char_to_int, int_to_char=self.int_to_char
            ),
        )

    def val_dataloader(self):
        """Create validation data loader."""
        assert self.val_dataset is not None

        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=CollateFn(
                char_to_int=self.char_to_int, int_to_char=self.int_to_char
            ),
        )
