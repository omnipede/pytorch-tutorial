from _8_word_representation._1_word2vec import download_training_data
from glove import Glove


def sample():
    """
    GloVe 학습 예제. 훈련 데이터는 Word2Vec 와 동일한 데이터를 사용한다.
    """

    training_data = download_training_data()

    t = Glove(training_data)

    pass
