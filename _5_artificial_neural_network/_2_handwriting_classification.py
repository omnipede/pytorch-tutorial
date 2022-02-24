from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import torch
from torch import nn


def sample():
    digits = load_digits()

    plt.show()

    X = torch.tensor(digits.data, dtype=torch.float32)
    Y = torch.tensor(digits.target, dtype=torch.int64)

    model = nn.Sequential(
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 10),
        nn.ReLU()
    )

    optimizer = torch.optim.Adam(model.parameters())
    cost = nn.CrossEntropyLoss()

    losses = []
    for epoch in range(100):
        predict = model(X)

        loss = cost(predict, Y)
        losses.append(loss)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if epoch % 10 == 0:
            print('Epoch {:4d}/{} Cost: {:.6f}'.format(
                epoch, 100, loss.item()
            ))

    plt.plot(losses)
    plt.show()

    pass
