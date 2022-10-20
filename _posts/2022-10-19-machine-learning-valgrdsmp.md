---
title: "교차검증 & 샘플링 & 그리드 서치"
excerpt: "Cross Validation, Grid Search, Sampling"

categories:
  - Machine Learning
  - Python
tags:
  - [Machine Learning, Python]

permalink: /machine-learning/valgrdsmp/

toc: true
toc_sticky: true

date: 2022-10-20
last_modified_at: 2022-10-20
---

## __교차검증(Cross Validation)__
- 모델 학습 시 데이터를 훈련용과 검증용으로 교차
- Train / Validation / Test
  - Train : 학습 셋
  - Validation : 검증 셋

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate

# 데이터 셋
wine = pd.read_csv('https://bit.ly/wine_csv_data')

# input / target 1차 분리
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

# train / valid / test 분리
train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=42
)
sub_input, val_input, sub_target, val_target = train_test_split(
    train_input, train_target, test_size=0.2, random_state=42
)

# 모형 만들고 평가 - valid데이터 평가
dt =  DecisionTreeClassifier(random_state = 42)
dt.fit(sub_input, sub_target)

print(dt.score(sub_input, sub_target))
print(dt.score(val_input, val_target))
#0.9971133028626413
#0.864423076923077

print(dt.score(test_input, test_target))
#0.8569230769230769

# 교차검증
scores = cross_validate(dt, train_input, train_target)

print(scores)
#{'fit_time': array([0.0121038 , 0.01054382, 0.01129436, 0.01120663, 0.01073456]), 
#'score_time': array([0.00164652, 0.00206113, 0.00140262, 0.00141835, 0.00147724]), 
#'test_score': array([0.86923077, 0.84615385, 0.87680462, 0.84889317, 0.83541867])}

scores['test_score'] # -> 샘플링 결과가 편향되어있기 때문에 값의 차이가 다 다르다
#array([0.86923077, 0.84615385, 0.87680462, 0.84889317, 0.83541867])

# 검증 데이터 score(테스트데이터 score)
np.mean(scores['test_score'])
# 0.855300214703487
```
