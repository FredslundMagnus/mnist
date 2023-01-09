from os.path import join

import hydra
import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

import wandb
from src.models.model import MyAwesomeModel

wandb.init(project="test-project", entity="fredslund")


def accuracy(model, data, labels):
    out = model(data)
    predicted, actual = out.argmax(axis=1), labels.argmax(axis=1)
    correct = predicted == actual
    return float(sum(correct) / 5000)


def load(
    train_data: str, train_label: str, test_data: str, test_label: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.load(join("..", "..", "..", "data", "processed", train_data)),
        torch.load(join("..", "..", "..", "data", "processed", train_label)),
        torch.load(join("..", "..", "..", "data", "processed", test_data)),
        torch.load(join("..", "..", "..", "data", "processed", test_label)),
    )


@hydra.main(config_path="../../config", config_name="config.yaml", version_base="1.1")
def train(cfg: DictConfig):
    model_cfg: dict = cfg.model_conf
    training_cfg: dict = cfg.training_conf
    wandb.config = {**model_cfg, **training_cfg}
    lr = training_cfg["lr"]
    epochs = training_cfg["epochs"]
    print("Training day and night")
    print(f"{lr = }")
    print(f"{epochs = }")

    model = MyAwesomeModel(model_cfg)

    wandb.watch(model)
    train_data, train_label, test_data, test_label = load(
        "train_data.tensor",
        "train_labels.tensor",
        "test_data.tensor",
        "test_labels.tensor",
    )
    print(train_data.shape, train_label.shape)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = MSELoss()
    dataset: Dataset = list(zip(train_data, train_label))  # type: ignore
    train_loader: DataLoader = DataLoader(dataset, batch_size=100, shuffle=True)

    history = [0] * epochs
    val = [0] * epochs
    for epoch in range(epochs):
        for data, label in train_loader:
            out = model(data)
            loss = loss_fn(out, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            history[epoch] += loss.detach().numpy()
            wandb.log({"loss": loss})
        val[epoch] = accuracy(model, test_data, test_label)
        print(epoch, history[epoch], val[epoch])
    torch.save(model, join("..", "..", "..", "models", "trained_model.pt"))
    torch.save(model, "trained_model.pt")
    plt.plot(range(len(history)), history)
    plt.savefig(join("..", "..", "..", "reports", "figures", "training_curve.png"))
    plt.savefig("training_curve.png")


if __name__ == "__main__":
    train()
