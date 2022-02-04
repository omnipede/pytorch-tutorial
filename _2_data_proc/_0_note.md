# Pytorch data 처리

```torch.utils.data``` 에 존재하는 ```Dataset``` , ```DataLoader``` 를 이용한 데이터 처리방법을 정리.

## 미니 배치

전체 훈련 데이터가 학습에 한번 사용된 주기를 ```epoch``` 라고 한다. 만약 데이터가 아주 클 경우 한번에 학습시키는게 불가능 하기 때문에  
전체 훈련 데이터를 여러 개로 나누어서 학습시키는데 나누어진 학습 데이터를 ```Minibatch``` 라고 하고 각 미니배치의 크기를 ```batch size```, 미니배치의 개수를 ```iteration``` 라 한다.

![스크린샷 2022-02-04 오후 3 16 21](https://user-images.githubusercontent.com/41066039/152481621-bd122bf4-1a51-4936-993f-2ee0a0d28bac.png)

위와 같은 자료구조를 쉽게 다룰 수 있도록 pytorch 는 ```Dataset```, ```DataLoader``` 를 제공한다.

## Pytorch 실습

* [대표 예제](./_1_mini_batch.py)
* [Dataset 을 상속 받아 커스텀 데이터셋 정의](./_2_custom_dataset.py)
