---
title: "머신러닝 기초"
excerpt: "머신 러닝 중요 메서드와 생선분류"

categories:
  - Machine Learning
  - Python
tags:
  - [Machine Learning, Python]

permalink: /machine-learning/basic/

toc: true
toc_sticky: true

date: 2022-10-17
last_modified_at: 2022-10-17
---
<br/><br/>
## 머신러닝 중요 메서드
<br/>
- fit() : 훈련시 사용하는 메서드, 두개의 데이터가 들어감.
    - 독립변수 : fish_data(길이, 몸무게)
    - 종속변수 : fish_target
- predict() : 예측할 때 사용
    - 새로운 데이터 : 독립변수만 추가
- score() : 모형의 성능 평가
    - 실무에서는 평가지표 함수를 사용!
- seed() : 초깃값이 같으면 동일한 난수를 뽑는다. 실험 재현성
- suffle : 주어진 배열을 랜덤하게 섞는다
<br/><br/>
## 생선 분류 문제
<br/>


### 데이터 구성
<br/>


```python
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

length = bream_length + smelt_length
weight = bream_weight + smelt_weight

fish_data = [[l, w] for l, w in zip(length, weight)]
fish_target = [1] * 35 + [0] * 14 # 라벨링 : 도미1 / 빙어0
```
<br/>


### 데이터 시각화
<br/>

```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize = (10,6))
ax.scatter(bream_length, bream_weight)
ax.scatter(smelt_length, smelt_weight)
ax.set_xlabel('length')
ax.set_ylabel('weight')

plt.show()
```
![a](/assets/images/posts_img/machine-learning-first/fish_1.png)
<br/><br/>
