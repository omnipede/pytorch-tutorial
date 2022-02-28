import torch.nn as nn


def sample():
    """
    Pytorch nn.Embedding() 을 이용한 임베딩 벡터 예제
    """

    # 훈련 데이터
    train_data = "you need to know how to code"

    # 단어 집합 생성
    word_set = set(train_data.split())

    vocab = {
        tkn: i + 2 for i, tkn in enumerate(word_set)
    }

    vocab['<unk>'] = 0
    vocab['<pad>'] = 1

    embedding_layer = nn.Embedding(
        # 임베딩할 단어 개수
        num_embeddings=len(vocab),
        # 임베딩할 벡터 차원
        embedding_dim=3,
        # 패딩 토큰의 인덱스. 해당 토큰에 대해서는 파라미터가 변하지 않음
        padding_idx=1
    )

    print(embedding_layer.weight)
