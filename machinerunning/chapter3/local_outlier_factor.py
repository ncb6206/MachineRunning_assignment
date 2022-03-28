import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

np.random.seed(42)  # 난수 설정
X_inliners = 0.3 * np.random.randn(100, 2) # 데이터 생성
X_inliners = np.r_[X_inliners + 2, X_inliners - 2]
X_outliers = np.random.uniform(low = -4, high = 4, size = (20, 2)) # 균등분포로부터 무작위 표본 추출
X = np.r_[X_inliners, X_outliers]   # 두 배열을 왼쪽에서 오른쪽으로 붙이기
n_outliers = len(X_outliers)    # 길이 측정
ground_truth = np.ones(len(X), dtype=int)   # int타입을 갖는 -1로 채워진 X길이 만큼의 어레이 반환
ground_truth[-n_outliers:] = -1

clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1) # 이상치 감지 모델 생성
y_pred = clf.fit_predict(X)
n_errors = (y_pred != ground_truth).sum()   # 받은 데이터 판단
X_scores = clf.negative_outlier_factor_ # 이상치 여부 판단

plt.title("Local Outlier Factor (LOF)")
plt.scatter(X[:, 0], X[:, 1], color='k', s=3, label='Data points')   # 산점도 X1,X2, 색깔, 크기 설정 후 표시
radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
# 산점도 X1,X2, 색깔, 크기 설정 후 표시
plt.scatter(X[:, 0], X[:, 1],s=1000 * radius, edgecolors='r', facecolors='none', label='Outlier scores')
#Matplotlib X,Y축 범위 지정
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.xlabel("prediction eroors: %d" % (n_errors))    # X축 이름
# plt에 범례와 크기 추가
legend = plt.legend(loc = 'upper left')
legend.legendHandles[0]._sizes = [10]
legend.legendHandles[1]._sizes = [20]
plt.show()