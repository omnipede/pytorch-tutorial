import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


def sample():
    """
    Epoch = 전체 훈련 데이터가 학습에 한 번 사용된 주기
    Iteration = 한번의 epoch 내에서 이루어지는 parameter (e.g. W, b) 업데이트 횟수.
    """

    x_train = torch.FloatTensor([[73, 80, 75],
                                 [93, 88, 93],
                                 [89, 91, 90],
                                 [96, 98, 100],
                                 [73, 66, 70]])
    y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

    dataset = TensorDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = nn.Linear(3, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

    nb_epochs = 20
    for epoch in range(nb_epochs + 1):
        for batch_idx, samples in enumerate(dataloader):
            x_train, y_train = samples

            prediction = model(x_train)
            cost = F.mse_loss(prediction, y_train)

            cost.backward()
            optimizer.step()
            optimizer.zero_grad()

            print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
                epoch, nb_epochs, batch_idx + 1, len(dataloader),
                cost.item()
            ))

    p = torch.FloatTensor([73, 80, 75])
    pred_y = model(p)
    print("훈련 후 입력이 73, 80, 75일 때의 예측값 :", pred_y)
