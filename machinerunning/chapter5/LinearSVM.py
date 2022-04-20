import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix

X, y = make_blobs(n_samples=250, centers=2, random_state=0, cluster_std = 0.85) # 클러스링 용 가상데이터 생성
y[y == 0] = -1  # 0값을 -1로 변환
y = y.astype(float) # 실수로 변경
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)   # x, y의 train/test 분리

svc = LinearSVC(C=0.5)  # C가 0.5, 1인 경우의 모델 설정 후 학습
svc.fit(X_train, y_train)

y_pred = svc.predict(X_train)   # x train의 결과 예측 후 y train과 예측 행렬 표시
print('training data = \n', confusion_matrix(y_train, y_pred))

y_pred = svc.predict(X_test)    # x test의 결과 예측 후 y test와 예측 행렬 표시
print('testing data = \n', confusion_matrix(y_test, y_pred))

def f(x, w, b, c=0):    # 함수의 모양 설정
    return (-w[0] * x - b + c) / w[1]

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='winter');    # 산점도 그리기
ax = plt.gca()  # 현재 Axes 객체 반환
xlim = ax.get_xlim()    # 현재 설정된 x축눈금 알아내기
w = svc.coef_[0]    # 클래스별 계수 설정
a = -w[0] / w[1]    # 계수의 기울기를 구함
xx = np.linspace(xlim[0], xlim[1])  # x축 눈금을 이용하여 그래프 설정
yy = a * xx - svc.intercept_[0] / w[1]  # y 절편 설정
plt.plot(xx, yy)    # 얻은 리스트를 이용해 그래프 그림
yy = a * xx - (svc.intercept_[0] - 1) / w[1]  # y 절편 설정
plt.plot(xx, yy, 'k--') # 얻은 리스트를 이용해 점선 그래프 그림
yy = a * xx - (svc.intercept_[0] + 1) / w[1]  # y 절편 설정
plt.plot(xx, yy, 'k--') # 얻은 리스트를 이용해 점선 그래프 그림
plt.xlim(-1,4)  # x 축의 범위 설정
plt.title('Linear SVM(C=0.5)') # 타이틀 설정 후 표시
plt.show()