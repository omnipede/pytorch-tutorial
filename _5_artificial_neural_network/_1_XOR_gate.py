import torch
from torch import nn
from torch import optim


def sample():
    """
    XOR gate 예제
    """

    X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = torch.FloatTensor([[0], [1], [1], [0]])

    model = nn.Sequential(
        nn.Linear(2, 10, bias=True),
        nn.Sigmoid(),
        nn.Linear(10, 10, bias=True),
        nn.Sigmoid(),
        nn.Linear(10, 10, bias=True),
        nn.Sigmoid(),
        nn.Linear(10, 1, bias=True),
        nn.Sigmoid()
    )

    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=1)

    for epoch in range(10001):
        hypo = model(X)
        cost = criterion(hypo, Y)
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()

        # 100의 배수에 해당되는 에포크마다 비용을 출력
        if epoch % 100 == 0:
            print(epoch, cost.item())
