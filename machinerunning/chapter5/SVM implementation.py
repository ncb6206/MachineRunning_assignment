import numpy as np
import cvxopt
from sklearn.datasets._samples_generator import make_blobs
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


class SVM:
    def fit(self, X, y):
        n_samples, n_features = X.shape  # 데이터 개수 속성 개수
        K = np.zeros((n_samples, n_samples))  # H = X^T X
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = np.dot(X[i], X[j])    # X[i]와 X[j]를 곱함
        H = cvxopt.matrix(np.outer(y, y) * K)
        f = cvxopt.matrix(np.ones(n_samples) * -1)  # f = -1 (1xN)
        B = cvxopt.matrix(y, (1, n_samples))  # B =y^T
        b = cvxopt.matrix(0.0)  # b > O
        A = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))  # -1 (NxN)
        a = cvxopt.matrix(np.zeros(n_samples))  # 0 (1xN)
        solution = cvxopt.solvers.qp(H, f, A, a, B, b)  # quadratic problem solver

        a = np.ravel(solution['x'])  # 라그랑주 승수
        sv = a > 1e-5  # 라그랑주 승수가 0보다 큰것
        ind = np.arange(len(a))[sv] # 배열 선언
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        self.b = 0  # 절편
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n], sv])
        self.b /= len(self.a)
        self.w = np.zeros(n_features)  # 가중치
        for n in range(len(self.a)):
            self.w += self.a[n] * self.sv_y[n] * self.sv[n]

    def project(self, X):   # X와 self.w를 곱한 값과 self.b를 더함
        return np.dot(X, self.w) + self.b

    def predict(self, X):   # 배열 원소의 부호 판별
        return np.sign(self.project(X)) 


X, y = make_blobs(n_samples=250, centers=2, random_state=0, cluster_std=0.60)   # 클러스링 용 가상데이터 생성
y[y == 0] = -1  # 0값을 -1로 변환
y = y.astype(float) # 실수로 변경
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)    # x, y의 train/test 분리

svm = SVM() #서포트 벡터 머신 설정 후 학습
svm.fit(X_train, y_train)


def f(x, w, b, c=0):    # 함수의 모양 설정
    return (-w[0] * x - b + c) / w[1]

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='winter') # 산점도 그리기

a0 = -2;
a1 = f(a0, svm.w, svm.b)  # W.X + b = O
b0 = 4;
b1 = f(b0, svm.w, svm.b)  # W.X + b = O
plt.plot([a0, b0], [a1, b1], 'k')   # 앞서 얻은 데이터를 이용해 그래프를 그림

a0 = -2;
a1 = f(a0, svm.w, svm.b, 1)  # W.X + b = 1
b0 = 4;
b1 = f(b0, svm.w, svm.b, 1)  # W.X + b = 1
plt.plot([a0, b0], [a1, b1], 'k--')  # 앞서 얻은 데이터를 이용해 그래프를 그림

a0 = -2;
a1 = f(a0, svm.w, svm.b, -1)  # W.X + b = -1
b0 = 4;
b1 = f(b0, svm.w, svm.b, -1)  # W.X + b = -1
plt.plot([a0, b0], [a1, b1], 'k--')  # 앞서 얻은 데이터를 이용해 그래프를 그림

y_pred = svm.predict(X_test)    # xtest 데이터를 이용해 예측값 얻음
print('training\n', confusion_matrix(y_test, y_pred))   #  ytest와 예측값으로 혼동행렬 출력
y_pred = svm.predict(X_test)    # xtest 데이터를 이용해 예측값 얻음
print('test\n', confusion_matrix(y_test, y_pred))   #  ytest와 예측값으로 혼동행렬 출력
# 타이틀 설정하고 표시
plt.title('SVM')
plt.show()