# 단어 표현 방법

## 원 핫 인코딩

단어 집합 (vocabulary) 의 크기와 동일한 차원을 갖는 벡터를 선언한 , 해당하는 단어에 1을 세팅하고 나머지에 0을 세팅하여 단어 하나를 표한하는 방법이다.
단, 이 표현방법은 메모리 낭비가 심하고, 단어 사이의 유사성을 파악하기 힘들다는 단점이 있다.


## 워드 임베딩

원 핫 인코딩과 같은 표현 방법을 희소 표현 (sparse representation) 라고 하는데 이와 반대되는 개념이 밀집 표현 (dense representation) 이다. 
단어를 밀집 표현으로 표현하는 것을 워드 임베딩이라고 하며 임베딩 결과를 임베딩 벡터라고 한다. 임베딩 방법으로 ```word2vec```, ```LSA```, ```FastText```, ```Glove``` 등이 있다.


## Word2Vec

밀집 표현 방법 중 하나로 단어의 의미를 다차원 공간에 벡터화하는 ```분산 표현``` 이 있다. 분산 표현은 '비슷한 위치에 등장한 단어는 비슷한 의미를 가진다' 라는 분포 가에 기반하며 
분산 표현을 위한 학습 방법중의 하나가 ```word2vec``` 다.


### CBOW (Continuous bag of words)

Word2Vec 방법 중의 하나로써 주변의 단어로 중간 단어를 예측하는 방법이다. 여기서, '주변 단어' 의 범위를 윈도우라고 한다.  
예를 들어 'Hello world foo bar' 에서 윈도우 크기가 1 이라면 'world' 라는 단어의 주변 단어는 'Hello', 'foo' 다.  

![스크린샷 2022-02-23 오후 2 56 17](https://user-images.githubusercontent.com/41066039/155268465-5f26a693-cf06-4fc5-b519-72cdb904108e.png)


CBOW 를 수행하기 위한 입력으로 주변 단어들의 원 핫 인코딩 벡터가 주어지고 각 원 핫 벡터는 가중치 W 와 곱해진 뒤, 평균으로 만든다. 
평균 벡터는 다시 가중치 W' 과 곱해져 softmax 활성화 함수를 통과하여 스코어 벡터 (y') 가 된다.

![스크린샷 2022-02-23 오후 3 38 01](https://user-images.githubusercontent.com/41066039/155271797-9614f50a-8f83-4b41-8900-bc859847aef1.png)

스코어 벡터를 실제 단어의 원 핫 벡터와 비교하며, 이 때 비용 함수로 cross entropy 함수를 사용한다.

Cross entropy 함수의 결과가 최소가 되는 방향으로 역전파 시켜 가중치를 업데이트하여 임베딩 벡터를 생성하는데 사용한다.


### Skip-gram

CBOW 와 달리 중간 단어로 주변 단어를 예측하는 방법이다. 
즉 CBOW 의 도식에서 원 핫 벡터가 여러개 가 아닌 하나가 입력으로 주어지며, 여러 개의 스코어 벡터가 출력된다.


## GloVe

### 윈도우 기반 동시 등장 행렬

단어 집합의 단어들을 행과 열에 배치하고 i 번째 단어의 윈도우 범위 내에서 k 번재 단어가 등장한 횟수를 (i, k) 에 기록한 행렬이다.


### 동시 등장 확률

동시 등장 확률 P(k | i) 는 동시 등장 행렬의 i 행의 모든 값을 더하여 분모로 삼고 (i, k) 값을 분자로 삼은 확률이다. 즉 i 번째 단어가 등장했을 때 k 번째 단어가 동시에 등장할 확률이다.


### 손실 함수

GloVe 방식의 손실 함수, 비용함수는 다음과 같이 정의된다.

![스크린샷 2022-02-23 오후 6 08 48](https://user-images.githubusercontent.com/41066039/155289578-d2532ee6-c3c9-4592-b2c3-a524fb0b5139.png)

V 는 단어 집합의 크기, X 는 동시 등장 행렬, w 는 중심 단어 임베딩, w' 은 주변 단어 임베딩, b 는 편향을 의미한다.


## Pytorch 에서 임베딩 벡터를 사용하는 방법

파이토치에서 임베딩 벡터를 사용할 때, 임베딩 레이어를 처음부터 학습시키는 방법과 미리 훈련된 임베딩 벡터를 가져오는 방법이 있다.

처음부터 임베딩 벡터를 학습시킬 때 ```nn.Embedding()``` 메소드를 사용하면 되고 

각 방법에 대한 상세 설명은 실습 코드 참조.

## Pytorch 실습

* [Word2Vec](./_1_word2vec.py)
* [GloVe](./_2_glove.py)
* [nn.Embedding() 예제](./_3_torch_nn_embedding.py)
* [미리 학습된 워드 임베딩 예제](./_4_pretrained_word_embedding.py)
