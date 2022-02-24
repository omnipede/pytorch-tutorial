# 인공 신경망

## 퍼셉트론

다수의 입력에 대해 하나의 출력을 발생시키는 알고리즘을 퍼셉트론이라고 한다. 수식으로 표현하면

```
if x1w1 + x2w2 ... + xnwn >= K then y := 1
else y := 0
```

여기서 상수 K 를 좌변으로 넘겨 bias 로 표현하면

```
if x1w1 + x2w2 ... + xnwn + b >= 0 then y := 1
else y := 0
```

위와 같이 계산 결과값이 특정 상수 이상인지를 기준으로 출력값을 결정하는 함수를 계단 함수라고 한다.
계단함수와 같은 함수를 ```f``` 라고 하면

```
y = f(w1x1 + w2x2 ... + wnxn + b)
```

로 표현할 수 있고 ```f``` 를 활성화 함수라고 한다. 활성화 함수는 계단함수 뿐만 아니라 시그모이드, [소프트맥스](../_4_softmax_regression/_0_note.md) 함수
등이 사용될 수 있다.

입력 ***x*** 에 대해서 출력 ***y*** 가 정해졌을 때 ***w*** 는 가중치라고 부른다.

## 단층 퍼셉트론

위와 같은 퍼셉트론을 단층 퍼셉트론이라고 한다. 단층 퍼셉트론은 입력층과 출력층 단 2개 층으로 이루어져 있다.

<img width="298" alt="스크린샷 2022-02-14 오후 6 03 10" src="https://user-images.githubusercontent.com/41066039/153832852-266b8eb9-3812-4ac7-8f5b-0759b3f2725b.png">

## 다층 퍼셉트론 

다층 퍼셉트론은 단층 퍼셉트론과 다르게 중간에 은닉층이 존재하는 퍼셉트론이다. 그리고 은닉층의 개수가 2개 이상이 되는 경우, 이를 ***심층신경망 (Deep Neural Network, DNN)*** 이라고 한다.

<img width="291" alt="스크린샷 2022-02-14 오후 6 04 11" src="https://user-images.githubusercontent.com/41066039/153832914-6e169911-7e17-4585-95ba-e57b5c86014d.png">

## Pytorch 실습
* [XOR gate](./_1_XOR_gate.py)
* [손글씨 분류](./_2_handwriting_classification.py)
* [MNIST 데이터 분류](./_3_MNIST_classification.py)
