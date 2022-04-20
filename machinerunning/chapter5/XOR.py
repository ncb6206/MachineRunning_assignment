import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from sklearn.svm import SVC

def plot_xor(X, y, model, title, xmin=-3, xmax=3, ymin=-3, ymax=3):
    # x와 y의 최소값, 최대값을 사용하여 격자 그리드 만들기
    XX, YY = np.meshgrid(np.arange(xmin, xmax, (xmax-xmin)/1000),
                         np.arange(ymin, ymax, (ymax-ymin)/1000))
    ZZ = np.reshape(model.predict(np.array([XX.ravel(), YY.ravel()]).T), XX.shape)  # 모델의 예측값을 재배열
    plt.contourf(XX, YY, ZZ, сmap=mpl.cm.Paired_r, alpha=0.5)   # 등고선 그리기
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='b', marker='o', label='class 1', s=50)   # 파란색 산점도 그리기
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='r', marker='s', label='class 0', s=50)   # 붉은색 산점도 그리기
    #x축, y축의 한계와 타이틀과 x,y축 라벨 표시
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x")

np.random.seed(0)   # 난수 설정
X_xor = np.random.randn(200, 2) # 가우시안 표준 정규 분포에서 난수 matrix array 생성
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)    # xor 배타적 논리합 설정
y_xor = np.where(y_xor, 1, 0)   # 1과 0인 것만 추출
plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], c='b', marker='o', label='class 1', s=50)   # 파란색 산점도 그리기
plt.scatter(X_xor[y_xor == 0, 0], X_xor[y_xor == 0, 1], c='r', marker='s', label='class 0', s=50)   # 빨간색 산점도 그리기
#그래프에 범례와 제목, x축, y축 추가 후 표시
plt.legend()
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("XOR problem")
plt.show()

svc = SVC(kernel="linear").fit(X_xor, y_xor)    # linear 커널 설정
polysvc = SVC(kernel="poly", degree=2, gamma=1, coef0=0).fit(X_xor, y_xor) # poly 커널 설정
rbfsvc = SVC(kernel="rbf").fit(X_xor, y_xor)    # rbf 커널 설정
sigmoidsvc = SVC(kernel="sigmoid", gamma=2, coef0=2).fit(X_xor, y_xor) # sigmoid 커널 설정

plt.figure(figsize=(10, 14))  # 윈도우 창 크기 설정
plt.subplot(411)    # nrows=4, ncols=1, index=1
plot_xor(X_xor, y_xor, svc, "SVC with linear kernel")   # linear 그래프 그리기
plt.subplot(412)    # nrows=4, ncols=1, index=2
plot_xor(X_xor, y_xor, polysvc, "SVC with ploynomial kernel")   # polysvc 그래프 그리기
plt.subplot(413)    # nrows=4, ncols=1, index=3
plot_xor(X_xor, y_xor, rbfsvc, "SVC with RBF kernel")   # rbfsvc 그래프 그리기
plt.subplot(414)    # nrows=4, ncols=1, index=4
plot_xor(X_xor, y_xor, sigmoidsvc, "SVC with sigmoid kernel")   # sigmoidsvc 그래프 그리기
# 그래프에 범례와 겹치지 않도록 여백 추가 후 표시
plt.tight_layout()
plt.legend()
plt.show()
