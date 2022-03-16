# SMOTE
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from matplotlib import pyplot
from numpy import where

# make_classification(): n_redundant (no. of redundant features)
#                        weights (portion of samples assigned to a class)
# 표본 데이터수 10000, 독립 변수의 수 2, 독립 변수 중 다른 독립 변수의 선형 조합으로 나타나는 성분의 수 0,
# 클래스 당 클러스터 수 1, 각 클래스에 할당된 표본 수 0.99, 랜덤 시드 설정한 가상의 분류 모형 데이터 생성
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
                           n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)

counter = Counter(y) # y 데이터의 항목 개수
print('generated data: %s' % counter)
# 받은 데이터를 이용하여 0, 1이 표본인 것들의 산점도를 표시
for label, _ in counter.items():
    row_ix = where(y == label)[0]
    pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()
# sampling strategy : |minority samples|/|majority samples| after resampling
# SMOTE기반으로 재샘플링 한 뒤 데이터 항목 설정
over = SMOTE(sampling_strategy=0.1)
X,y = over.fit_resample(X,y)
counter = Counter(y)
print('SMOTE-based oversampled data: %s' % counter)

# SMOTE기반으로 오버샘플링한 데이터를 이용하여 0, 1이 표본인 것들의 산점도를 표시
for label, _ in counter. items():
    row_ix = where(y == label)[0]
    pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()
