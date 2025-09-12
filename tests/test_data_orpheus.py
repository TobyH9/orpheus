from pathlib import Path
import pytest

import numpy as np

from orpheus.data.data import TinyShakeDataset

TESTS_DATA_PATH = Path(__file__).parent / "data" / "test_tiny_shakespeare.txt"

# Shared fixtures
@pytest.fixture(scope="module")
def dataset() -> TinyShakeDataset:
    return TinyShakeDataset(
        data_path=str(TESTS_DATA_PATH),
        split="train",
        block_size = 8,)


class TestTinyShakeDataset:

    def test_train_dataset_initialization(self, dataset: TinyShakeDataset):
        "Test that the TinyShakeDataset can be initialised"

        assert dataset is not None
        assert dataset.data_path == str(TESTS_DATA_PATH)
        assert dataset.split == "train"

    def test_dataset_getitem(self, dataset: TinyShakeDataset):
        """Test that dataset can retrieve individual items."""
        # Test getting an item
        item = dataset[0]
        assert isinstance(item, tuple)
        assert isinstance(dataset[0][0], np.ndarray)
        assert isinstance(dataset[0][1], np.ndarray)
        assert len(dataset[0][0]) == 8
        assert len(dataset[0][1]) == 8


