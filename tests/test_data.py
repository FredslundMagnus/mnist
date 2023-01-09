import os
from os.path import join

import pytest
import torch

from tests import _PATH_DATA


def load(
    train_data: str, train_label: str, test_data: str, test_label: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.load(join(_PATH_DATA, "processed", train_data)),
        torch.load(join(_PATH_DATA, "processed", train_label)),
        torch.load(join(_PATH_DATA, "processed", test_data)),
        torch.load(join(_PATH_DATA, "processed", test_label)),
    )


train_data, train_label, test_data, test_label = load(
    "train_data.tensor",
    "train_labels.tensor",
    "test_data.tensor",
    "test_labels.tensor",
)


@pytest.mark.skipif(
    not os.path.exists(join(_PATH_DATA, "processed")), reason="Data files not found"
)
def test_datas():
    assert (
        len(train_data) == 40000
    ), "Train-Dataset did not have the correct number of samples"
    assert (
        len(train_label) == 40000
    ), "Train-Dataset did not have the correct number of samples"
    assert (
        len(test_data) == 5000
    ), "Test-Dataset did not have the correct number of samples"
    assert (
        len(test_label) == 5000
    ), "Test-Dataset did not have the correct number of samples"
    assert tuple(train_data.shape[1:]) == (
        28 * 28,
    ), "Train-Dataset did not have the correct shape"
    assert tuple(test_data.shape[1:]) == (
        28 * 28,
    ), "Test-Dataset did not have the correct shape"
    assert all(
        [int(a.item()) != 0 for a in train_label.sum(axis=0)]
    ), "Not all labels are present in the train dataset"
    assert all(
        [int(a.item()) != 0 for a in test_label.sum(axis=0)]
    ), "Not all labels are present in the test dataset"
