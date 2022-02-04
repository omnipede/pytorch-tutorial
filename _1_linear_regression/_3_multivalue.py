import torch
import torch.optim as optim


def sample():
    """
    입력값 하나에 출력값이 하나가 맵핑되는게 아니라 여러 입력값에 대해 하나의 출력값이 맵핑되는 경우다.
    이 때 W, b 에 상수 대신 행렬을 넣는다.
    """

    x_train = torch.FloatTensor([
        [73, 80, 75],
        [93, 88, 93],
        [89, 91, 90],
        [96, 98, 100],
        [73, 66, 70]
    ])

    y_train = torch.FloatTensor([
        [152],
        [185],
        [180],
        [196],
        [142]
    ])

    W = torch.zeros((3, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    optimizer = optim.SGD([W, b], lr=1e-5)

    nb_epochs = 1000
    for epoch in range(nb_epochs + 1):

        hypo = x_train.matmul(W) + b
        cost = ((hypo - y_train) ** 2).mean()

        cost.backward()
        optimizer.step()
        optimizer.zero_grad()

        if epoch % 100 == 0:
            print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6f}'.format(
                epoch, nb_epochs, hypo.squeeze().detach(), cost.item()
            ))

    pass
