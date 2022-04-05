import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()     # 아이리스 데이터 불러오기
X = iris.data # 데이터 입력
y = iris.target # 데이터 출력

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  # x, y의 train/test 분리
scaler = StandardScaler()  # Standardization 평균 0 / 분산 1
# 교차검증시
scaler.fit(X_train)
X_train = scaler.transform(X_train) #속성값 정규화
X_test = scaler.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=5) #KNN 분류기
classifier.fit(X_train, y_train)  # 분류한 데이터 학습
y_pred = classifier.predict(X_test)  # X데이터 예측값 저장

print(confusion_matrix(y_test, y_pred)) # 혼동행렬 생성
print(classification_report(y_test, y_pred)) # 분류 결과값 출력

error = [ ]
for i in range(1, 40): # K값의 범위 (1..40)로 두고 KNN 분류기 이용해 데이터 학습과 예측값 저장 error에 평균 요소 추가
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        error.append(np.mean(pred_i != y_test))

# 생성한 데이터에 그래프와 점을 그리고 제목, x,y이름 추가 후 표시
plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()