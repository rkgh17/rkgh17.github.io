---
title: "합성곱 신경망(CNN)"
excerpt: "CNN을 이용한 Fashion mnist모델 학습 과정의 이해"

categories:
  - Deep Learning
tags:
  - [Deep Learning]

permalink: /deep-learning/cnn/

toc: true
toc_sticky: true

date: 2022-10-31
last_modified_at: 2022-10-31
---

# CNN을 이용한 Fashion mnist모델 학습 과정의 이해

---

## 시작하기 전에

### 합성곱 신경망이란

- 다차원 배열 데이터를 처리하도록  아래의 층으로 구성된 신경망
    - 입력층, 합성곱층, 풀링층, 완전연결층, 출력층
- 합성곱층과 풀링층을 거치며 이미지의 주요 특성 벡터를 추출
- 추출된 주요 특성 벡터들은 완전연결층을 거치며 1차원 벡터로 변환됨
- 마지막으로 출력층에서 활성화 함수를 사용하여 최종 결과 출력

### 데이터 전처리

- 데이터 다차원화

```python
# 데이터 불러오기
from tensorflow import keras
from sklearn.model_selection import train_test_split

(train_input, train_target), (test_input, test_target) = \
  keras.datasets.fashion_mnist.load_data()

# 데이터 체크
train_input.shape, train_target.shape, test_input.shape, test_target.shape
# ((60000, 28, 28), (60000,), (10000, 28, 28), (10000,))

# 정규화
train_scaled = train_input.reshape(-1,28,28,1) / 255.0

# 정규화 데이터 체크
train_scaled.shape
# (60000, 28, 28, 1)

# 데이터셋 분리
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state = 42
)

# 분리 데이터 체크
train_scaled.shape, val_scaled.shape
# ((48000, 28, 28, 1), (12000, 28, 28, 1))
```

---

## 합성곱 신경망 만들기

![1](/assets/images/posts_img/deep-learning-cnn/cnn.png)

```python
model = keras.Sequential()

model.add(keras.layers.Conv2D(32,                          # 32개의 필터
                              kernel_size = 3,             # 커널사이즈 3*3
                              activation='relu',           # 활성화 함수 relu 
                              padding='same',              # 패딩을 사용
                              input_shape=(28, 28, 1)))    # 입력층 셋팅
model.add(keras.layers.MaxPooling2D(2))

model.add(keras.layers.Conv2D(64,
                              kernel_size = 3,
                              activation='relu',
                              padding='same'))
model.add(keras.layers.MaxPooling2D(2))

# 1차원
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dropout(0.4)) # Dropout : 노드의 일부를 Drop -> 과적합 방지
model.add(keras.layers.Dense(10, activation='softmax'))

model.summary()
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  conv2d (Conv2D)             (None, 28, 28, 32)        320       
                                                                 
#  max_pooling2d (MaxPooling2D  (None, 14, 14, 32)       0         
#  )                                                               
                                                                 
#  conv2d_1 (Conv2D)           (None, 14, 14, 64)        18496     
                                                                 
#  max_pooling2d_1 (MaxPooling  (None, 7, 7, 64)         0         
#  2D)                                                             
                                                                 
#  flatten (Flatten)           (None, 3136)              0         
                                                                 
#  dense (Dense)               (None, 100)               313700    
                                                                 
#  dropout (Dropout)           (None, 100)               0         
                                                                 
#  dense_1 (Dense)             (None, 10)                1010      
                                                                 
# =================================================================
# Total params: 333,526
# Trainable params: 333,526
# Non-trainable params: 0
# _________________________________________________________________

# 모델 컴파일
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics='accuracy')

# 모델 시각화
keras.utils.plot_model(model)
```

![2](/assets/images/posts_img/deep-learning-cnn/layer.png)

### 모델 저장 및 학습

```python
# 모델 저장
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-cnn-model.h5',
                                                save_best_only = True)
# 과적합 방지
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2,
                                                  restore_best_weights=True)

# 모델 학습
history = model.fit(train_scaled, train_target, epochs=50,
                    validation_data = (val_scaled, val_target),
                    callbacks=[checkpoint_cb, early_stopping_cb])
															# 모든 에포크마다 모델 저장 / 과적합 방지
# Epoch 1/50
# 1500/1500 [==============================] - 17s 4ms/step - loss: 0.5243 - accuracy: 0.8132 - val_loss: 0.3264 - val_accuracy: 0.8808
# Epoch 2/50
# 1500/1500 [==============================] - 6s 4ms/step - loss: 0.3475 - accuracy: 0.8758 - val_loss: 0.2821 - val_accuracy: 0.8947
# Epoch 3/50
# 1500/1500 [==============================] - 6s 4ms/step - loss: 0.3001 - accuracy: 0.8911 - val_loss: 0.2521 - val_accuracy: 0.9068
# Epoch 4/50
# 1500/1500 [==============================] - 6s 4ms/step - loss: 0.2686 - accuracy: 0.9023 - val_loss: 0.2427 - val_accuracy: 0.9097
# Epoch 5/50
# 1500/1500 [==============================] - 6s 4ms/step - loss: 0.2454 - accuracy: 0.9106 - val_loss: 0.2402 - val_accuracy: 0.9108
# Epoch 6/50
# 1500/1500 [==============================] - 6s 4ms/step - loss: 0.2263 - accuracy: 0.9171 - val_loss: 0.2297 - val_accuracy: 0.9130
# Epoch 7/50
# 1500/1500 [==============================] - 6s 4ms/step - loss: 0.2095 - accuracy: 0.9224 - val_loss: 0.2270 - val_accuracy: 0.9165
# Epoch 8/50
# 1500/1500 [==============================] - 6s 4ms/step - loss: 0.1919 - accuracy: 0.9286 - val_loss: 0.2321 - val_accuracy: 0.9149
# Epoch 9/50
# 1500/1500 [==============================] - 6s 4ms/step - loss: 0.1812 - accuracy: 0.9324 - val_loss: 0.2211 - val_accuracy: 0.9210
# Epoch 10/50
# 1500/1500 [==============================] - 6s 4ms/step - loss: 0.1668 - accuracy: 0.9376 - val_loss: 0.2226 - val_accuracy: 0.9208
# Epoch 11/50
# 1500/1500 [==============================] - 6s 4ms/step - loss: 0.1554 - accuracy: 0.9408 - val_loss: 0.2247 - val_accuracy: 0.9245
# 설정해둔 early_stopping_cb으로 인해 학습 일찍 종료
```

### 모델 평가

```python
import matplotlib.pyplot as plt

# 모델 평가
model.evaluate(val_scaled, val_target)
#375/375 [==============================] - 1s 3ms/step - loss: 0.2211 - accuracy: 0.9210
#[0.22110480070114136, 0.9210000038146973]

# 그래프로 시각화하기
def eval_graph(history):  
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.legend(['train', 'validation'])
  plt.show()

eval_graph(history)
```

![3](/assets/images/posts_img/deep-learning-cnn/graph.png)

### 저장된 모델 불러와보기

```python
model2 = keras.models.load_model('/content/best-cnn-model.h5')
model2.summary()
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  conv2d (Conv2D)             (None, 28, 28, 32)        320       
                                                                 
#  max_pooling2d (MaxPooling2D  (None, 14, 14, 32)       0         
#  )                                                               
                                                                 
#  conv2d_1 (Conv2D)           (None, 14, 14, 64)        18496     
                                                                 
#  max_pooling2d_1 (MaxPooling  (None, 7, 7, 64)         0         
#  2D)                                                             
                                                                 
#  flatten (Flatten)           (None, 3136)              0         
                                                                 
#  dense (Dense)               (None, 100)               313700    
                                                                 
#  dropout (Dropout)           (None, 100)               0         
                                                                 
#  dense_1 (Dense)             (None, 10)                1010      
                                                                 
# =================================================================
# Total params: 333,526
# Trainable params: 333,526
# Non-trainable params: 0
# _________________________________________________________________

# 불러온 모델 평가
model2.evaluate(val_scaled, val_target)
#375/375 [==============================] - 1s 3ms/step - loss: 0.2211 - accuracy: 0.9210
#[0.22110480070114136, 0.9210000038146973]
```