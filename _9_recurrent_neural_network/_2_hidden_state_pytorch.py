import torch
import torch.nn as nn


def sample():
    """
    Pytorch 를 이용한 RNN 구현 예제
    """
    input_size = 5
    hidden_size = 8

    # (배치 크기, 시점의 수, 매 시점마다 들어가는 입력)
    inputs = torch.Tensor(1, 10, input_size)

    cell = nn.RNN(
        input_size,
        hidden_size,
        # 은닉층의 개수
        num_layers=2,
        # 입력 텐서 첫번재 차원이 배치 크기임을 선언
        batch_first=True,
        # 양방향 여부
        bidirectional=True
    )

    # 전체 은닉 상태, 마지막 시점 은닉 상태
    hidden_states, hidden_state = cell(inputs)

    # (배치 크기, 시점의 수, 은닉 상태의 크기 * 2) --> 양방향이기 때문에 2를 곱한다.
    print(hidden_states.shape)

    # (층의 개수 * 2, 배치 크기, 은닉 상태의 크기) --> 양방향이기 때문에 2를 곱한다.
    print(hidden_state.shape)


