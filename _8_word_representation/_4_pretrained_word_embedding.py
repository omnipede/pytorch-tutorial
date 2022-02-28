import torch.nn
from torchtext.legacy import data, datasets
from torchtext.vocab import GloVe


def sample():
    """
    이미 학습된 임베딩 레이어를 사용하는 방법
    """

    TEXT = data.Field(sequential=True, batch_first=True, lower=True)
    LABEL = data.Field(sequential=False, batch_first=True)

    print("Splitting IMDB data ...")
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL, root="~/.torchtext/data")
    print("IMDB data is splitted.")

    # 단어 집합 생성
    TEXT.build_vocab(
        train_data,
        # 사전 훈련된 임베딩 벡터
        vectors=GloVe('6B', dim=300, cache="~/.torchtext/vectors"),
        # 단어 집합 최대 크기
        max_size=10000,
        # 최소 등장 빈도수
        min_freq=10
    )

    # 임베딩 벡터 예시
    index_of_a_word = TEXT.vocab.stoi['this']
    embedding_vector_of_a_word = TEXT.vocab.vectors[index_of_a_word]

    print(embedding_vector_of_a_word)

    # Pre train 된 벡터로부터 embedding layer 를 구축
    embedding_layer = torch.nn.Embedding.from_pretrained(TEXT.vocab.vectors, False)

    print(embedding_layer(torch.LongTensor([10])))
