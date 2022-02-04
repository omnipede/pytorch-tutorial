import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MultivariableLinearRegressionModel3By1(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)


def sample():
    """
    NN module 을 상속받은 클래스를 이용한 예제
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

    model = MultivariableLinearRegressionModel3By1()

    optimizer = optim.SGD(model.parameters(), lr=1e-5)

    nb_epochs = 2000
    for epoch in range(nb_epochs + 1):

        prediction = model(x_train)
        cost = F.mse_loss(prediction, y_train)

        cost.backward()
        optimizer.step()
        optimizer.zero_grad()

        if epoch % 100 == 0:
            # 100번마다 로그 출력
            print('Epoch {:4d}/{} Cost: {:.6f}'.format(
                epoch, nb_epochs, cost.item()
            ))
