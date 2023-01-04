from os.path import join

import hydra
import matplotlib.pyplot as plt
import torch
from model import MyAwesomeModel
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import os

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
    model_cfg = cfg.model_conf
    training_cfg = cfg.training_conf
    lr = training_cfg["lr"]
    epochs = training_cfg["epochs"]
    print("Training day and night")
    print(f"{lr = }")
    print(f"{epochs = }")

    model = MyAwesomeModel(model_cfg)
    train_data, train_label, test_data, test_label = load(
        "train_data.tensor",
        "train_labels.tensor",
        "test_data.tensor",
        "test_labels.tensor",
    )
    print(train_data.shape, train_label.shape)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = MSELoss()
    train_loader = DataLoader(
        list(zip(train_data, train_label)), batch_size=100, shuffle=True
    )

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
        val[epoch] = accuracy(model, test_data, test_label)
        print(epoch, history[epoch], val[epoch])
    torch.save(model, join("models", "trained_model.pt"))
    plt.plot(history)
    plt.savefig(join("reports", "figures", "training_curve.png"))



if __name__ == "__main__":
    train()
