import spacy
import nltk
from nltk import word_tokenize

# 주의) en module 을 다운로드 받아야 한다
# $ python -m spacy download en
spacy_en = spacy.load('en_core_web_sm')
nltk.download('punkt')


def sample():
    """
    Tokenize 예시.
    """
    en_text = "A Dog Run back corner near spare bedrooms"
    spacy_tokenized = [tok.text for tok in spacy_en.tokenizer(en_text)]
    print(spacy_tokenized)

    nltk_tokenized = word_tokenize(en_text)
    print(nltk_tokenized)
