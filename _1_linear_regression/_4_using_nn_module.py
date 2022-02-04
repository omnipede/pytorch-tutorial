import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def sample():
    """
    파이토치에서 제공하는 nn 함수를 이용해서 선형 모델 구현
    """

    x_train = torch.FloatTensor([
        [1],
        [2],
        [3]
    ])

    y_train = torch.FloatTensor([
        [2],
        [4],
        [6]
    ])

    # Input: 1차원, Output: 1차원
    model = nn.Linear(1, 1)

    optimizer = optim.SGD(model.parameters(), lr=0.01)

    nb_epochs = 2000
    for epoch in range(nb_epochs + 1):

        prediction = model(x_train)

        # 평균 제곱 오차 함수
        cost = F.mse_loss(prediction, y_train)

        cost.backward()
        optimizer.step()
        optimizer.zero_grad()

        if epoch % 100 == 0:
            # 100번마다 로그 출력
            print('Epoch {:4d}/{} Cost: {:.6f}'.format(
                epoch, nb_epochs, cost.item()
            ))