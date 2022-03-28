from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
import numpy as np
from matplotlib import pyplot as plt

X, Y = make_moons(n_samples=1000, noise=0.05)   # 초승달 모양 클러스터 두 개 형상의 데이터 생성

plt.title('Half moons')
plt.scatter(X[:,0], X[:,1])  # 산점도 X1,X2, 크기 설정 후 표시
plt.show()

dbs = DBSCAN(eps=0.1)   # DBSCAN으로 군집화한 모델 생성 및 학습
Z = dbs.fit_predict(X)

colormap = np.array(['r', 'b']) # 색깔을 위한 배열 선언
plt.scatter(X[:,0], X[:,1], c=colormap[Z])  # 산점도 X1,X2,색깔 설정 후 표시
plt.title('DBSCAN for half moons')
plt.show()