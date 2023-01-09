from torch import nn


class MyAwesomeModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        hidden_layers: list[int] = cfg["hidden_layers"]
        layers: list[nn.Module] = []
        for i, o in zip([28 * 28, *hidden_layers], [*hidden_layers, 10]):
            layers.append(nn.Linear(i, o, dtype=float))
            layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers[:-1])

    def forward(self, out):
        return self.model(out)
