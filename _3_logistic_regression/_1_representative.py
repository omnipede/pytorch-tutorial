import torch


def sample():
    """
    로지스틱 회귀 대표 예제
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

    W = torch.zeros((2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    optimizer = torch.optim.SGD([W, b], lr=1)

    total_epochs = 1000
    for epoch in range(total_epochs + 1):

        hypo = torch.sigmoid(x_train.matmul(W) + b)
        cost = -(y_train * torch.log(hypo) + (1 - y_train) * torch.log(1 - hypo)).mean()

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print('Epoch {:4d}/{} Cost: {:.6f}'.format(
                epoch, total_epochs, cost.item()
            ))
