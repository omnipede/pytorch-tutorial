import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init as init
from torch.utils.data import DataLoader


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        # 특성 맵 크기 계산 공식 참조
        # Input: 28 * 28 * 1
        # Conv -> 28 * 28 * 32
        # Pool -> 14 * 14 * 32
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Input: 14 * 14 * 32
        # Conv -> 14 * 14 * 64
        # Pool -> 7 * 7 * 64
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Linear(7 * 7 * 64, 10, bias=True)

        init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)


def sample():
    batch_size = 50

    mnist_train = dsets.MNIST(root='MNIST_data/',  # 다운로드 경로 지정
                              train=True,  # True를 지정하면 훈련 데이터로 다운로드
                              transform=transforms.ToTensor(),  # 텐서로 변환
                              download=True)

    mnist_test = dsets.MNIST(root='MNIST_data/',  # 다운로드 경로 지정
                             train=False,  # False를 지정하면 테스트 데이터로 다운로드
                             transform=transforms.ToTensor(),  # 텐서로 변환
                             download=True)

    data_loader = DataLoader(dataset=mnist_train,
                             batch_size=batch_size,
                             shuffle=True,
                             drop_last=True)

    total_batch = len(data_loader)
    print('총 배치의 수 : {}'.format(total_batch))

    model = CNN()

    criterian = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    training_epochs = 15
    for epoch in range(training_epochs):

        avg_cost = 0

        for X, Y in data_loader:
            hypo = model(X)
            cost = criterian(hypo, Y)
            cost.backward()
            optimizer.step()
            optimizer.zero_grad()
            avg_cost += cost / total_batch

        print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

    with torch.no_grad():
        X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float()
        Y_test = mnist_test.test_labels

        prediction = model(X_test)
        correct_prediction = torch.argmax(prediction, 1) == Y_test
        accuracy = correct_prediction.float().mean()
        print('Accuracy:', accuracy.item())
