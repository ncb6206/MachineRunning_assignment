import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture

n_samples = 600 # 샘플 수 설정
np.random.seed(0)   # 랜덤 씨드 설정
# 가우시안 표준 정규 분포에서 얻은 난수 matrix array 와 난수 array 배열 합쳐서 shifted_gaussian 변수 설정
shifted_gaussian = np.random.randn(n_samples, 2) + np.array([15, 30])
C1 = np.array([[0., -0.7], [1.5, .7]]) # C1 배열 설정
# 가우시안 표준 정규 분포에서 얻은 난수 matrix array 와 난수 array 배열 합쳐서 shifted_gaussian1 변수 설정
stretched_gaussian1 = np.dot(np.random.randn(n_samples, 2), C1)+ np.array([-5, -25])
C2 = np.array([[0.5, 1.7], [-1.5, 0.5]])  # C2 배열 설정
# 가우시안 표준 정규 분포에서 얻은 난수 matrix array 와 난수 array 배열 합쳐서 shifted_gaussian2 변수 설정
stretched_gaussian2 = np.dot(np.random.randn(n_samples, 2), C2)+ np.array([-15, 5])
X_train = np.vstack([shifted_gaussian, stretched_gaussian1, stretched_gaussian2])  # 세 배열을 위에서 아래로 붙이기
# Matplotlib의 X,Y축 범위 지정
plt.xlim(-50,50)
plt.ylim(-50,50)
plt.scatter(X_train[:, 0], X_train[:, 1], .8, color='r')  # 산점도 X1,X2, 색깔 설정 후 표시
plt.show()

clf = mixture.GaussianMixture(n_components=3, covariance_type='full') # 가우시안 모델 생성 후 학습
clf.fit(X_train)
# X, Y 1차원 배열 범위 설정 후 배열 생성
x = np.linspace(-50., 50.)
y = np.linspace(-50., 50.)
X, Y = np.meshgrid(x,y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -clf.score_samples(XX) # 데이터에 대한 로그 밀도 모델 평가
Z = Z.reshape(X.shape)  # 배열과 차원 변형

# 앞서 설정 한 데이터를 표현할 등고선 설정
plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10))
plt.scatter(X_train[:, 0], X_train[:, 1], .8, color='r')    # 산점도 X1,X2, 색깔 설정 후 표시
plt.title('Mixture of Gaussians')
plt.show()