from os.path import join

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


def test_datas():
    assert len(train_data) == 40000
    assert len(train_label) == 40000
    assert len(test_data) == 5000
    assert len(test_label) == 5000
    assert tuple(train_data.shape[1:]) == (28 * 28,)
    assert tuple(test_data.shape[1:]) == (28 * 28,)
    assert all([int(a.item()) != 0 for a in train_label.sum(axis=0)])
    assert all([int(a.item()) != 0 for a in test_label.sum(axis=0)])
