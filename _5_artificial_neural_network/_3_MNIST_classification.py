import numpy as np
import torch
from torch import nn, optim
from sklearn.datasets import fetch_openml
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


def sample():

    print("Download MNIST data ...")
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    print("Download is finished.")

    X = mnist.data / 255
    Y = mnist.target.astype(np.int8)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/7, random_state=0)

    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    Y_train = torch.LongTensor(Y_train)
    Y_test = torch.LongTensor(Y_test)

    ds_train = TensorDataset(X_train, Y_train)
    ds_test = TensorDataset(X_test, Y_test)

    loader_train = DataLoader(ds_train, batch_size=64, shuffle=True)
    loader_test = DataLoader(ds_test, batch_size=64, shuffle=False)

    model = nn.Sequential(
        nn.Linear(28 * 28, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(100):
        model.train()

        for data, target in loader_train:
            predicted = model(data)

            loss = loss_function(predicted, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print("epoch{}：완료\n".format(epoch))
