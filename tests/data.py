from tests import _PATH_DATA
from os.path import join
import torch

def load(
    train_data: str, train_label: str, test_data: str, test_label: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.load(join("..", "..", "..", "data", "processed", train_data)),
        torch.load(join("..", "..", "..", "data", "processed", train_label)),
        torch.load(join("..", "..", "..", "data", "processed", test_data)),
        torch.load(join("..", "..", "..", "data", "processed", test_label)),
    )

train_data, train_label, test_data, test_label = load(
        "train_data.tensor",
        "train_labels.tensor",
        "test_data.tensor",
        "test_labels.tensor",
    )
assert len(train_data) == 40000
assert len(train_label) == 4000
assert len(test_data) == 5000
assert len(test_label) == 5000
# assert that each datapoint has shape [1,28,28] or [728] depending on how you choose to format
# assert that all labels are represented