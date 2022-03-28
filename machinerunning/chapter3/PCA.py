from sklearn.decomposition import PCA
from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

iris = datasets.load_iris() # 아이리스 데이터 불러오기
print(list(iris.keys()))    # 데이터의 키 리스트 출력
X = iris["data"][:,0:4] # 아이리스 데이터를 배열로 받음
label = iris["target"]  # 아이리스 데이터를 타켓시킴

pca = PCA(n_components = 2) # 주성분을 몇개로 할지 결정후 PCA 모델 생성 및 학습
X2D = pca.fit_transform(X)

for i,j in enumerate(np.unique(label)): # 빨강, 파랑, 녹색 세가지 인자 설정 후 표시
    plt.scatter(X2D[label == j, 0], X2D[label == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.legend()    # 그래프에 범례 추가 후 표시
plt.show()
