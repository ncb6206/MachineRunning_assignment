from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from matplotlib import pyplot as plt

X, Y = make_blobs(n_samples=2000, n_features=2, centers=8, cluster_std=2.0) # 인수 설정한 뒤 가상 데이터 생성
plt.scatter(X[:,0], X[:,1], s=4) # 산점도 X1,X2, 크기 설정 후 표시
plt.title('Generated Data')
plt.show()

# 모델 생성 및 학습
Z = AgglomerativeClustering(n_clusters=8, linkage='complete')
P = Z.fit_predict(X)
colormap = np.array(['r', 'g', 'b', 'k', 'y', 'c', 'm', 'orange'])  # 색깔을 위한 배열 선언
plt.scatter(X[:,0], X[:,1], s=4, c=colormap[P]) # 산점도 X1,X2, 크기, 색깔 설정 후 표시
plt.title('Clustering results')
plt.show()