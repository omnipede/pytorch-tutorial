import numpy as np
import torch
from torch import nn
from torch import optim


def sample():
    """
    다대다 RNN 구현 예제.
    apple 을 입력 받으면 pple! 라는 결과를 출력하도록 함.
    """

    input_str = 'apple'
    label_str = 'pple!'

    # 문자 집합 생성
    char_vocab = sorted(list(set(input_str + label_str)))
    vocab_size = len(char_vocab)
    input_size = vocab_size
    hidden_size = 5
    output_size = 5

    alpha = 0.1

    # 각 문자 별로 정수 부여
    ctoi = dict((c, i) for i, c in enumerate(char_vocab))
    itoc = dict((i, c) for i, c in enumerate(char_vocab))

    x_data = [ctoi[c] for c in input_str]
    y_data = [ctoi[c] for c in label_str]

    # 배치 차원 추가
    x_data = [x_data]
    y_data = [y_data]

    # 원 핫 인코딩
    x_one_hot = [np.eye(vocab_size)[x] for x in x_data]

    X = torch.FloatTensor(x_one_hot)
    Y = torch.LongTensor(y_data)

    # 모델 선언
    net = Net(input_size, hidden_size, output_size)

    # 비용함수
    criterion = nn.CrossEntropyLoss()

    # 최적화 함수
    optimizer = optim.Adam(net.parameters(), alpha)

    for i in range(100):
        outputs = net(X)

        loss = criterion(outputs.view(-1, input_size), Y.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # 가장 큰 인덱스를 선택
        # ex) [-2.3899, -5.0211,  1.0820,  7.4773,  8.1834] 에서 4를 선택
        result = outputs.data.numpy().argmax(axis=2)
        # 결과 문자열 확인
        result_str = ''.join([itoc[c] for c in np.squeeze(result)])
        print(i, "loss: ", loss.item(), "prediction: ", result, "true Y: ", y_data, "prediction str: ", result_str)


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()

        # RNN 셀
        self.rnn = nn.RNN(input_size, hidden_size, output_size)

        # 출력층
        self.fc = torch.nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, x):
        """
        RNN 은닉 층과 출력층을 연결
        """
        x, _status = self.rnn(x)
        x = self.fc(x)
        return x
