import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
np.random.seed(0) # 난수 설정
X = np.sort(5*np.random.rand(40,1), axis = 0) # 정렬된 배열의 복사본 생성, 원본은 변경 없음
T = np.linspace(0,5,500)[:, np.newaxis] # 수평축의 간격 만들기
y = np.sin(X).ravel() # X의 sin 삼각함수값 다차원 배열을 1차원 배열로 변경

y[::5] += 1*(0.5 - np.random.rand(8)) #잡음 추가

n_neighbors = 5 # 근접이웃 설정
for i, weights in enumerate(['uniform','distance']): # KNN 기반 회귀를 이용해 데이터 학습과 예측값 저장
  knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
  y_ = knn.fit(X,y).predict(T)

  plt.subplot(2, 1, i+1)  # 여러개의 그래프와 축 공유
  plt.scatter(X, y, color='darkorange',label='data')  # 산점도 그리기
  plt.plot(T,y_,color='navy', label='rediction') # 그래프 그리기
  plt.axis('tight')  # 모든 데이터를 볼수 있을 정도로 축의 범위를 충분히 크게 설정
  #그래프에 범례와 제목 추가 후 표시
  plt.legend()
  plt.title("KNeighbrosRegressor (k=%i, weights='%s')"%(n_neighbors, weights))

plt.tight_layout()
plt.show()