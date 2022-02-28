import re
import urllib.request
from lxml import etree

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

from gensim.models import Word2Vec


def sample():
    """
    Word2vec 실습
    """

    # 훈련 데이터 다운로드
    result = download_training_data()

    # Word2Vec model 훈련
    model = Word2Vec(
        # 대상 단어 집합
        sentences=result,
        # 임베딩 벡터 크기
        vector_size=100,
        # 윈도우 크기
        window=5,
        # 최소 단어 빈도수
        min_count=5,
        # 학습시 사용할 워커 수
        workers=4,
        # Skip-gram 방식 사용 여부
        sg=0
    )

    # "man" 과 가장 유사한 단어를 찾는다.
    similar_to_man = model.wv.most_similar("man")
    print(similar_to_man)


def download_training_data():
    """
    훈련 데이터 다운로드 방법
    """
    print("Downloading training data")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/09.%20Word%20Embedding/dataset/ted_en-20160408.xml",
        filename="ted_en-20160408.xml")

    # XML 파싱
    targetXML = open('ted_en-20160408.xml', 'r', encoding='UTF8')
    target_text = etree.parse(targetXML)

    parse_text = '\n'.join(target_text.xpath('//content/text()'))

    content_text = re.sub(r'\([^)]*\)', '', parse_text)

    # 데이터 크기가 너무 커서 오래걸리기 때문에 자른다.
    content_text = content_text[:200000]

    nltk.download('punkt')

    # 문장 토큰화
    print(f"Sentence tokenizing ... {len(content_text)}")
    sent_text = sent_tokenize(content_text)

    normalized_text = []
    for string in sent_text:
        tokens = re.sub(r"[^a-z0-9]+", " ", string.lower())
        normalized_text.append(tokens)

    # 단어 토큰화
    print("Word tokenizing ...")
    result = [word_tokenize(sentence) for sentence in normalized_text]

    return result
