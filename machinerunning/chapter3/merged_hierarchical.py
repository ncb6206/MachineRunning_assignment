from sklearn.datasets import make_blobs
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
import numpy as np
from matplotlib import pyplot as plt

X, Y = make_blobs(n_samples=25, n_features=2, centers=3, cluster_std=1.5)   # 인수 설정한 뒤 가상 데이터 생성
colormap = np.array(['r', 'g', 'b'])    # 빨강, 파랑, 녹색의 색깔 설정
plt.scatter(X[:,0], X[:,1], c=colormap[Y])  # 산점도 X1,X2 설정
plt.title('Generated Data')
plt.show()  # 가상 데이터 표시

Xdist = pdist(X, metric='euclidean') # 거리 행렬 계산
Z = linkage(Xdist, method='ward') # ward 연결을 통한 계층적 군집화
Zd = dendrogram(Z)  # 덴드로그램 그리고 표시
plt.title('Dendrogram')
plt.show()