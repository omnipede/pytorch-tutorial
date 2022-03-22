from torchtext.legacy import data, datasets


def sample():
    """
    다대일 RNN 예제. 영화 리뷰를 읽고, 해당 리뷰가 긍정적인지, 부정적인지 분류함.
    """

    TEXT = data.Field(sequential=True, batch_first=True, lower=True)
    LABEL = data.Field(sequential=False, batch_first=True)

    print("Splitting train_data and test_data")
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL, root="~/.torchtext/data")

    pass
