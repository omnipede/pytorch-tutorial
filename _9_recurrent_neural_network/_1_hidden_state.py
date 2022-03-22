import numpy as np


def sample():
    """
    Python 으로 hidden state 를 구현한 예제
    """

    timesteps = 10

    # 입력 차원
    input_size = 4

    # 은닉 상태의 크기
    hidden_size = 8

    # 10 x 4 텐서
    inputs = np.random.random((timesteps, input_size))

    hidden_state_t = np.zeros((hidden_size,))

    print(hidden_state_t)

    # 8x4 텐서
    Wx = np.random.random((hidden_size, input_size))
    # 8x8 텐서
    Wh = np.random.random((hidden_size, hidden_size))
    # 8x1 텐서
    b = np.random.random((hidden_size,))

    for xt in inputs:
        hidden_state_after = np.tanh(np.dot(Wx, xt) + np.dot(Wh, hidden_state_t) + b)
        hidden_state_t = hidden_state_after

    print(hidden_state_t)
