import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets._samples_generator import make_blobs
from sklearn.cluster import KMeans

X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)    # 인수 설정한 뒤 가상 데이터 생성
plt.scatter(X[:,0], X[:,1], s=4)    # 산점도 X1,X2, 크기 설정 후 표시
plt.title('Generated Data')
plt.show()

# k-means 데이터 생성 및 학습
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(X)
colormap = np.array(['c', 'g', 'b', 'm'])   # 색깔을 위한 배열 선언
plt.scatter(X[:,0], X[:,1], s=4, c=colormap[pred_y])    # 산점도 X1,X2, 크기, 색깔 설정 후 표시
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='red')
plt.title('Clustering Results with Centers')
plt.show()
