import pytest
import torch

from src.models.model import MyAwesomeModel


def test_model_shape():
    model = MyAwesomeModel(cfg={"hidden_layers": [200, 100, 50]})
    input = torch.rand(1, 28 * 28, dtype=float)
    output = model(input)
    assert output.shape == (1, 10)


def test_model_input_warnings():
    model = MyAwesomeModel(cfg={"hidden_layers": [200, 100, 50]})
    with pytest.raises(RuntimeError):
        model(torch.rand(1, dtype=float))
