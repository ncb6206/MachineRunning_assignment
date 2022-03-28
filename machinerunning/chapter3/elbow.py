from matplotlib import pyplot as plt
from sklearn.datasets._samples_generator import make_blobs
from sklearn.cluster import KMeans

X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)   # 인수 설정한 뒤 가상 데이터 생성
plt.scatter(X[:, 0], X[:, 1])   # 산점도 X1,X2 설정 후 표시
plt.show()

SSEs = []
for i in range(1, 11):
    # k-means 데이터 생성 및 학습
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit_predict(X)
    SSEs.append(kmeans.inertia_)  # SSE 값 저장

plt.plot(range(1, 11), SSEs, marker='o')    # 얻은 k-means 데이터 값 그래프로 표시
plt.show()