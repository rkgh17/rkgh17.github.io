---
title: "차원축소 - PCA"
excerpt: "주성분분석 내용정리"

categories:
  - Deep Learning
tags:
  - [Deep Learning]

permalink: /deep-learning/pca/

toc: true
toc_sticky: true

date: 2022-10-24
last_modified_at: 2022-10-24
---

---

# 차원축소 - PCA(Principal component analysis)

---

## 비정형 데이터

지정된 방식으로 정리되지 않은 정보 (이미지, 비디오, 텍스트 문장이나 문서, 음성 데이터)를 **비정형데이터**라고 하며 이러한 데이터들은 매우 많은 특성(feature)들을 가지고 있다

데이터 분류 측면에서 데이터의 차원이 크면(특성이 많으면) 학습 속도가 느릴 뿐만 아니라 성능이 좋지 않기 때문에 **차원 축소**라는 것을 하게된다

차원을 축소하는 방법 중 하나인 **주성분 분석**(PCA, Principal Component Analysis)의 진행과정과 코드에 대해 알아보자

---

## 데이터 셋 - 시각화

```python
import numpy as np
import matplotlib.pyplot as plt

# 데이터셋 로드
!wget https://bit.ly/fruits_300_data -O fruits_300.npy
fruits = np.load('fruits_300.npy')

fruits.shape
# (300, 100, 100) : 100*100차원인 300장의 고사진들

# 시각화
fig, ax = plt.subplots(1,3, figsize=(12,5))
ax[0].imshow(fruits[0], cmap='gray_r')
ax[1].imshow(fruits[100], cmap='gray_r')
ax[2].imshow(fruits[200], cmap='gray_r')
plt.show()
```

![a](/assets/images/posts_img/deep-learning-pca/download.png)

```python
# 3차원 데이터들 → 2차원 or 1차원으로 변경
fruits_2d = fruits.reshape(-1, 100*100)
fruits_2d.shape
# (300, 10000)
```
---

## sklearn을 통한 PCA 변환

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
pca.fit(fruits_2d)

print(pca.components_.shape)
# (50, 10000)

# 50장 데이터에 100*100 차원으로 변환
pca.components_.reshape(-1, 100, 100).shape
# (50, 100, 100)

fruits_2d.shape
#(300, 10000)

fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape)
#(300, 50)
```

---

## 시각화

```python
import matplotlib.pyplot as plt

# 시각화 함수 정의
def draw_fruits(arr, ratio=1):
    n = len(arr)    # n은 샘플 개수입니다
    # 한 줄에 10개씩 이미지를 그립니다. 샘플 개수를 10으로 나누어 전체 행 개수를 계산합니다. 
    rows = int(np.ceil(n/10))
    # 행이 1개 이면 열 개수는 샘플 개수입니다. 그렇지 않으면 10개입니다.
    cols = n if rows < 2 else 10
    fig, axs = plt.subplots(rows, cols, 
                            figsize=(cols*ratio, rows*ratio), squeeze=False)
    for i in range(rows):
        for j in range(cols):
            if i*10 + j < n:    # n 개까지만 그립니다.
                axs[i, j].imshow(arr[i*10 + j], cmap='gray_r')
            axs[i, j].axis('off')
    plt.show()

# pca변환 시각화
draw_fruits(pca.components_.reshape(-1, 100, 100))
```

![b](/assets/images/posts_img/deep-learning-pca/download (1).png)

1000개의 특성을 50개 주성분으로 표현해도 데이터의 의미(사과모양의 구)는 보전

→ 압축해도 잘 쓸수 있다

---

## 원본 데이터 재구성

그렇다면 다시 복원 시켜도 원래의 데이터와 비슷할까?

```python
# pca복원 : pca.inverse
fruits_inverse = pca.inverse_transform(fruits_pca)
print(fruits_inverse.shape)
# (300, 10000)

# 복원된 값 시각화
fruits_reconstruct = fruits_inverse.reshape(-1,100,100)
for i in [0,100,200]:
  draw_fruits(fruits_reconstruct[i:i+100])
  print('\n')
```

![c](/assets/images/posts_img/deep-learning-pca/download (2).png)

![d](/assets/images/posts_img/deep-learning-pca/download (3).png)

![e](/assets/images/posts_img/deep-learning-pca/download (4).png)

→ 원래의 데이터와 완벽히 같지는 않지만 어느정도 보임

---

## 설명된 분산(Explained Variance)

- 원본 데이터의 분산을 얼마나 잘 나타내는지 기록한 값

```python
print(np.sum(pca.explained_variance_ratio_))
# 0.9215488639090053 -> 92%만큼

plt.plot(pca.explained_variance_ratio_)
plt.show()
# 주성분 n_components = 10으로해도 50이랑 그닥 그렇게 다르지는 않다
```

![f](/assets/images/posts_img/deep-learning-pca/download (5).png)

---

## 다른 알고리즘과 함께 사용하기

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

lr = LogisticRegression()

# 특성 3개로 압축시
target = np.array([0]*100 + [1]*100 + [2]*100)

scores = cross_validate (lr, fruits_2d, target)
print(np.mean(scores['test_score'])) # 99.6%
print(np.mean(scores['fit_time']))   # 1초

scores = cross_validate(lr, fruits_pca, target)
print(np.mean(scores['test_score'])) # 98.6%
print(np.mean(scores['fit_time']))   # 0.05초

```
