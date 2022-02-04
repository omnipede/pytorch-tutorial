import torch
import torch.optim as optim
from torch import Tensor


def sample():
    """
    옵티마이저 알고리즘을 통해 적절한 W, b 를 찾는 과정을 ML 에서 학습이라고 부른다.
    """

    x_train = torch.FloatTensor([[1], [2], [3]])
    y_train = torch.FloatTensor([[2], [4], [6]])

    W = torch.zeros(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    # SGD = 경사 하강법의 일종
    optimizer = optim.SGD([W, b], lr=0.01)

    nb_epochs = 1999
    for epoch in range(nb_epochs + 1):
        # 가설 선언
        # h(x) = wx + b
        hypo: Tensor = x_train * W + b
        # 가설과 실제 값 사이의 오차를 나타내는 비용함수 cost(W, b)
        cost: Tensor = ((hypo - y_train) ** 2).mean()

        # 미분
        cost.backward()
        optimizer.step()
        # 주의) Optimizer 는 미분 결과를 누적시키기 때문에 기울기를 0으로 초기화시킨다.
        optimizer.zero_grad()

        if epoch % 100 == 0:
            print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
                epoch, nb_epochs, W.item(), b.item(), cost.item()
            ))

    pass
