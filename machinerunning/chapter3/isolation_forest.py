import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
n_samples = 500 # 샘플 수 설정
# outlier와 inlier 변수 설정
outliers_fraction = 0.05
n_outliers = int(outliers_fraction * n_samples)
n_inliers = n_samples - n_outliers
blobs_params = dict(random_state=0, n_samples=n_inliers, n_features=2)  # 키=값 형식으로 딕셔너리 만듦 
X = make_blobs(centers=[[0, 0], [4, 3]],    #클러스링 용 가상데티어 생성
               cluster_std=0.5,
               **blobs_params)[0]
rng = np.random.RandomState(42) # 난수 설정
X = np.concatenate([X, rng.uniform(low=-6, high=6, size=(n_outliers, 2))], axis=0) # 2차원 배열을 위 아래 방향으로 연결

iForest = IsolationForest(n_estimators=20, verbose=2) # isolation forest 모델 생성 후 학습
iForest.fit(X)
# 학습 한 모델을 이용해 산점도 X1,X2, 색깔 설정 후 표시
pred = iForest.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=pred, cmap='RdBu')
plt.show()

pred_scores = -1 * iForest.score_samples(X) # 다른 모델을 이용해 데이터 저장
plt.scatter(X[:, 0], X[:, 1], c=pred_scores, cmap='RdBu')   # 산점도 X1,X2, 색깔 설정 후 표시
plt.colorbar(label='Simplified Anomaly Score')
plt.show()
