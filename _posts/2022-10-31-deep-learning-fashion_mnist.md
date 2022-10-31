---
title: "머신러닝, 인공신경망, 심층신경망"
excerpt: "머신러닝, 인공신경망, 심층신경망을 이용한 Fashion mnist 모델 학습하기"

categories:
  - Deep Learning
tags:
  - [Deep Learning]

permalink: /deep-learning/fashion_mnist/

toc: true
toc_sticky: true

date: 2022-10-31
last_modified_at: 2022-10-31
---

# 머신러닝, 인공신경망, 심층신경망을 이용한 Fashion mnist 모델 학습하기

---

## 시작하기 전에

### 데이터 분석

```python
# 데이터 불러오기
from tensorflow import keras

# train/test 분류
(train_input, train_target), (test_input, test_target) = \
	keras.datasets.fashion_mnist.load_data()

# 데이터 구성
train_input.shape , train_target.shape, test_input.shape, test_target.shape
# 의미 : 6만장의 train 이미지 / 만장의 test 이미지 / 크기는 28*28
```

### 데이터 구성 시각화

```python
import matplotlib.pyplot as plt
import numpy as np

label_results = [train_target[i] for i in range(10)]
print(np.unique(train_target, return_counts=True))
# [9, 0, 0, 3, 0, 2, 7, 2, 5, 5]
# 의미 : 아래의 각각의 데이터들은 위의 라벨값을 가짐

# 각 레이블마다 이미지의 갯수 확인
print(np.unique(train_target, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000]))
# 의미 : 각각 레이블마다 6천장의 이미지 데이터

fig, ax = plt.subplots(1, 10, figsize=(10,10))
for i in range(10):
  ax[i].imshow(train_input[i], cmap='gray_r')
  ax[i].axis('off')

plt.show()
```

![1](/assets/images/posts_img/deep-learning-fashion_mnist/fm.png)

---

## 머신러닝

### SGDClassifier로 이미지 분류

- [https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)

```python
from sklearn.model_selection import cross_validate
from sklearn.linear_model import SGDClassifier

# 데이터 정규화
train_scaled = train_input / 255.0 # -> 픽셀 0~255 를 0~1로 변환
train_scaled = train_scaled.reshape(-1, 28*28)

print(train_scaled.shape)
# (60000, 784)

# 모델 생성 및 훈련
sc = SGDClassifier(loss = 'log', max_iter = 5, random_state=42)
scores =cross_validate(sc, train_scaled, train_target, n_jobs=-1)

# 모델 평가
print(np.mean(scores['test_score']))
# 0.8195666666666668
```

---

## 인공신경망

- 입력층 : 784개의 뉴런 포함
- 출력층 : 10개의 뉴런 포함
    - 활성화 함수 : softmax
- 손실함수 : sparse_categorical_crossentropy

```python
from sklearn.model_selection import train_test_split
from tensorflow import keras

train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42
)

train_scaled.shape, val_scaled.shape, train_target.shape, val_target.shape
#((48000, 784), (12000, 784), (48000,), (12000,))

# 모델 생성 및 컴파일
output_dense = keras.layers.Dense(10, activation='softmax', input_shape=(784,))
model = keras.Sequential(output_dense)

model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')

# 모델 훈련
model.fit(train_scaled, train_target, epochs = 5)
# Epoch 1/5
# 1500/1500 [==============================] - 3s 2ms/step - loss: 0.6054 - accuracy: 0.7950
# Epoch 2/5
# 1500/1500 [==============================] - 3s 2ms/step - loss: 0.4775 - accuracy: 0.8394
# Epoch 3/5
# 1500/1500 [==============================] - 3s 2ms/step - loss: 0.4556 - accuracy: 0.8464
# Epoch 4/5
# 1500/1500 [==============================] - 2s 2ms/step - loss: 0.4439 - accuracy: 0.8524
# Epoch 5/5
# 1500/1500 [==============================] - 3s 2ms/step - loss: 0.4374 - accuracy: 0.8550
# <keras.callbacks.History at 0x7f02360a3f90>

# 모델 평가
model.evaluate(val_scaled, val_target)
#375/375 [==============================] - 1s 3ms/step - loss: 0.4499 - accuracy: 0.8496
#[0.44990962743759155, 0.8495833277702332]
```

---

## 심층 신경망

- 은닉층 하나 (200노드)
    - 활성화 함수 : relu

```python
from tensorflow import keras

# 은닉층 하나 추가 : 200개
hidden_dense = keras.layers.Dense(200,activation='relu', input_shape=(784,))
output_dense = keras.layers.Dense(10, activation='softmax')

model = keras.Sequential([hidden_dense,output_dense])

model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
model.fit(train_scaled, train_target, epochs = 5)
# Epoch 1/5
# 1500/1500 [==============================] - 6s 4ms/step - loss: 0.5201 - accuracy: 0.8141
# Epoch 2/5
# 1500/1500 [==============================] - 5s 3ms/step - loss: 0.3843 - accuracy: 0.8622
# Epoch 3/5
# 1500/1500 [==============================] - 5s 3ms/step - loss: 0.3507 - accuracy: 0.8748
# Epoch 4/5
# 1500/1500 [==============================] - 5s 4ms/step - loss: 0.3313 - accuracy: 0.8822
# Epoch 5/5
# 1500/1500 [==============================] - 7s 4ms/step - loss: 0.3177 - accuracy: 0.8875
# 375/375 [==============================] - 1s 2ms/step - loss: 0.3675 - accuracy: 0.8758

# 모델 평가
model.evaluate(val_scaled, val_target)
# [0.36753201484680176, 0.8758333325386047]
```

- 은닉층 둘 (200노드, 100노드)
    - 활성화 함수 : relu

```python
from tensorflow import keras

# 입력층
hidden_dense1 = keras.layers.Dense(200,activation='relu', input_shape=(784,))
hidden_dense2 = keras.layers.Dense(100,activation='relu')
output_dense = keras.layers.Dense(10, activation='softmax')

model = keras.Sequential([hidden_dense1,hidden_dense2,output_dense])

model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
model.fit(train_scaled, train_target, epochs = 5)
# Epoch 1/5
# 1500/1500 [==============================] - 7s 5ms/step - loss: 0.5118 - accuracy: 0.8157
# Epoch 2/5
# 1500/1500 [==============================] - 7s 5ms/step - loss: 0.3858 - accuracy: 0.8614
# Epoch 3/5
# 1500/1500 [==============================] - 7s 4ms/step - loss: 0.3643 - accuracy: 0.8703
# Epoch 4/5
# 1500/1500 [==============================] - 6s 4ms/step - loss: 0.3490 - accuracy: 0.8771
# Epoch 5/5
# 1500/1500 [==============================] - 6s 4ms/step - loss: 0.3410 - accuracy: 0.8818
# 375/375 [==============================] - 1s 2ms/step - loss: 0.3976 - accuracy: 0.8625

model.evaluate(val_scaled, val_target)
# [0.3976423442363739, 0.862500011920929]
```

무턱대고 은닉층을 추가한다 하여 모델의 성능이 좋아지란 보장은 없다.

### 은닉층을 추가하는 여러 방법

- 방법1

```python
# 객체생성
model = keras.Sequential()

model.add(keras.layers.Dense(100, activation = 'relu',input_shape=(784,)))
model.add(keras.layers.Dense(10, activation = 'softmax'))

model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
model.fit(train_scaled, train_target, epochs = 5)
model.evaluate(val_scaled, val_target)
```

- 방법2

```python
model = keras.Sequential()

model.add(keras.layes.Flatten(input_shape=(28,28)))
model.add(keras.layers.Dense(100, activation = 'relu'))
model.add(keras.layers.Dense(10, activation = 'softmax'))

model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
model.fit(train_scaled, train_target, epochs = 5)
model.evaluate(val_scaled, val_target)
```