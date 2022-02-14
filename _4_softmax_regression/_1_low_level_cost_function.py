import torch
import torch.nn.functional as F


def sample():

    # 임의의 벡터 z 에 대해 softmax 를 적용해본다.
    z = torch.randn(3, 5, requires_grad=True)
    hypothesis = F.softmax(z, dim=1)
    print(hypothesis)

    # y 는 [0, 2, 1] 와 같은 벡터
    y = torch.randint(5, (3, )).long()

    # 모든 원소가 0 인 3 * 5 벡터를 만든다.
    y_one_hot = torch.zeros_like(hypothesis)

    # y 를 unsqueeze 하면 [0], [2], [1] 처럼 된다.
    # unsqueeze 가 알려주는 위치에 1을 세팅한다.
    y_one_hot.scatter_(1, y.unsqueeze(1), 1)

    # 비용함수
    cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()

    pass
