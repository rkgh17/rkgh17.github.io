---
title: "Logistic regression"
excerpt: "로지스틱 회귀 맛보기"

categories:
  - Machine Learning
  - Python
tags:
  - [Machine Learning, Python]

permalink: /machine-learning/Logistic/

toc: true
toc_sticky: true

date: 2022-10-19
last_modified_at: 2022-10-19
---


## _Fishes_

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#물고기 7종류 - 5가지 특성 데이터셋
fish = pd.read_csv('https://bit.ly/fish_csv_data')

#
fig, ax = plt.subplots(figsize=(10,6))
sns.scatterplot(data = fish, x='Length', y='Weight', hue = 'Species')
plt.show()
```
![a](/assets/images/posts_img/machine-learning-4/fishes.png)

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# 전처리 - 변수 행렬로 변환하기 (데이터프레임으론 머신러닝 x)
# 독립변수
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
#종속변수
fish_target = fish['Species'].to_numpy()

#데이터 세트 분리
train_input, test_input, train_target, test_target = train_test_split(
    fish_input, fish_target, random_state=42
)
#train_input.shape, test_input.shape, train_target.shape, test_target.shape

#표준화
ss = StandardScaler() # 표준점수
ss.fit(train_input)

train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# KNN 머신 러닝으로 확률 예측
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)

proba = kn.predict_proba(test_scaled[:5])
# 소수점 4자리
print(np.round(proba, decimals=4))
```
![b](/assets/images/posts_img/machine-learning-4/KNN.png)

- 위 데이터셋과 같이 특성이 많으면, KNN 모델로는 특성 간의 관계나 디테일한 확률을 알기 힘듬.
- 로지스틱 회귀는 특성 간의 관계를 보여주고 특정 결과의 확률을 계산한다

<br/>

# __Logistic Regression__
- 여러 특성(이벤트)가 있을 경우 확률을 결정하는 데 사용되는 통계 모델

### 로지스틱 회귀 맛보기
- 확률값 구하기
    - z 값 구하기
    - 시그모이드 함수에 z값 대입 -> 확률값 계산
    - 확률값이 매우 큰 음수 --> 0
    - 확률값이 매우 큰 양수 --> 1
 

```python
import matplotlib.pyplot as plt

#인위로 z값 설정
z = np.arange(-5, 5, 0.1)
# 시그모이드 함수
phi = 1/ (1+np.exp(-z))

#시각화
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(z,phi)
ax.set_xlabel('z')
ax.set_ylabel('phi')

plt.show()
```
![c](/assets/images/posts_img/machine-learning-4/logi.png)

## _로지스틱 회귀로 이진 분류 수행하기_

- 현재 데이터 : train_scaled & test_scaled
- 현재 데이터에는 물고기들의 이름이 없기 때문에
- train_target, test_target과 Boolean Indexing 수행
- __Boolean Indexing__ : True, False 값을 활용함

```python
from sklearn.linear_model import LogisticRegression
from scipy.special import expit

# 주어진 데이터 Boolean indexing
bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
bream_smelt_indexes

#new data
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]

train_bream_smelt.shape, target_bream_smelt.shape # ((33, 5), (33,))

lr = LogisticRegression()
lr.fit(train_bream_smelt,target_bream_smelt)

#z값 함수
decisions= lr.decision_function(train_bream_smelt[:5])
print(decisions) # [-6.02927744  3.57123907 -5.26568906 -4.24321775 -6.0607117 ]

#시그모이드 함수
print(expit(decisions)) # [0.00240145 0.97264817 0.00513928 0.01415798 0.00232731]

#로지스틱 회귀로 다중 분류 수행하기
#규제 추가. ridge, lasso의 alpha와 반대개념
lr = LogisticRegression(C = 20, max_iter = 1000)
lr.fit(train_scaled,train_target)

print(lr.score(train_scaled, train_target)) # 0.9327731092436975
print(lr.score(test_scaled, test_target))   # 0.925

```
