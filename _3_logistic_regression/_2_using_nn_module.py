
import torch
import torch.nn as nn
import torch.nn.functional as F


def sample():
    """
    Pytorch nn 모듈을 이용해서 리팩토링한 예제
    """
    x_train = torch.FloatTensor([
        [1, 2],
        [2, 3],
        [3, 1],
        [4, 3],
        [5, 3],
        [6, 2]
    ])

    y_train = torch.FloatTensor([
        [0],
        [0],
        [0],
        [1],
        [1],
        [1]
    ])

    model = nn.Sequential(
        nn.Linear(2, 1),
        nn.Sigmoid()
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=1)

    total_epochs = 1000
    for epoch in range(total_epochs + 1):

        hypo = model(x_train)
        cost = F.binary_cross_entropy(hypo, y_train)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print('Epoch {:4d}/{} Cost: {:.6f}'.format(
                epoch, total_epochs, cost.item()
            ))
