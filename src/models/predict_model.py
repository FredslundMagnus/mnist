from os.path import join

import click
import numpy as np
import torch


@click.group()
def cli():
    pass


def load(test_data: str, test_label: str) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.load(join("data", "processed", test_data)), torch.load(
        join("data", "processed", test_label)
    )


def accuracy(model, data, labels):
    out = model(data)
    predicted, actual = out.argmax(axis=1), labels.argmax(axis=1)
    correct = predicted == actual
    return float(sum(correct) / 5000)


@click.command()
@click.argument("model_checkpoint")
def predict(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    model = torch.load(model_checkpoint)
    test_data, test_label = load("test_data.tensor", "test_labels.tensor")
    print(f"Accuracy: {accuracy(model, test_data, test_label):%}")
    data: dict[int, np.ndarray] = np.load(
        "data/example_images.npy", allow_pickle=True
    ).item()
    data: torch.Tensor = torch.stack(
        tuple([torch.as_tensor(img) for img in data.values()])
    )
    print(data.shape)
    result = model(data.reshape(10, -1)).argmax(axis=1)
    print("Prediction:", [i.item() for i in result])


cli.add_command(predict)

if __name__ == "__main__":
    cli()
