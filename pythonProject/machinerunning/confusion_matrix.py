import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

# 실제값과 예측값 각각 설정
y_true = ["positive", "negative", "negative", "positive", "positive","positive", "negative"]
y_pred = ["positive", "negative", "positive", "positive", "negative","positive", "positive"]

# 실제값과 예측값의 혼동 행렬 실행 후 정확도 평가
cm = confusion_matrix(y_true, y_pred)
print('confusion matrix')
print(cm)
a = accuracy_score(y_true, y_pred)
print('accuracy: ', a)

# 실제값과 예측값 재설정 후 정확도 평가
y_pred = [0, 5, 2, 4]
y_true = [0, 1, 2, 3]
a = accuracy_score(y_true, y_pred)
print('accuracy: ', a)
