import urllib.request
import pandas as pd
from torchtext.legacy import data
from torchtext.legacy.data import TabularDataset, Iterator


def sample():
    """
    Torchtext 예제
    """

    # IMDB 리뷰 데이터 다운로드
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/LawrenceDuan/IMDb-Review-Analysis/master/IMDb_Reviews.csv",
        filename="IMDb_Reviews.csv")

    df = pd.read_csv('IMDb_Reviews.csv', encoding='latin1')

    # 훈련데이터와 테스트데이터로 나누어서 csv 로 저장
    train_df = df[:25000]
    test_df = df[25000:]

    train_df.to_csv("train_data.csv", index=False)
    test_df.to_csv("test_data.csv", index=False)

    # Torchtext 필드 정의
    TEXT = data.Field(
        # 시퀀스 데이터 여부
        sequential=True,
        # 단어 집합 생성 여부
        use_vocab=True,
        # 어떤 토큰화 함수를 사용할지
        tokenize=str.split,
        # 영문자를 소문자로 만들지
        lower=True,
        # 문자열 데이터를 배치로 만들 때, 배치 크기를 텐서의 맨 앞에 놓을 지 여부
        # 예를 들어 batch size 가 5 이고 문자열의 길이가 20일 때 미니 배치 하나는 [5, 20] 텐서다.
        batch_first=True,
        # 최대 허용길이. Padding 기준.
        fix_length=20)

    # 리뷰 점수
    LABEL = data.Field(
        sequential=False,
        use_vocab=False,
        batch_first=False,
        # 레이블 데이터 여부
        is_target=True
    )

    # Train data, Test data loading
    train_data, test_data = TabularDataset.splits(
        path="./",
        train="train_data.csv",
        test="test_data.csv",
        format="csv",
        # 첫번째 컬럼 (review) 를 text 라고 부르고 TEXT 필드에 맵핑함
        # 두번째 컬럼 (sentiment) 를 label 라고 부르고 LABEL 필드에 맵핑함
        fields=[('text', TEXT), ('label', LABEL)],
        # 데이터 첫 줄 무시
        skip_header=True
    )

    # 단어 집합 만들기
    TEXT.build_vocab(train_data, min_freq=10, max_size=10000)

    # print(TEXT.vocab.stoi)

    # 데이터 로더 만들기
    batch_size = 5
    train_loader = Iterator(dataset=train_data, batch_size=batch_size)
    test_loader = Iterator(dataset=test_data, batch_size=batch_size)

    # Batch iteration example
    for batch in train_loader:
        print(batch.text)
