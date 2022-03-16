from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot

def evaluate_model(cv):
    # 표본 데이터수 100, 독립 변수의 수 20, 독립 변수 중 종속 변수와 상관 관계가 있는 성분의 수 15, 
    # 독립 변수 중 다른 독립 변수의 선형 조합으로 나타나는 성분의 수 5, 랜덤 시드 설정한 가상의 분류 모형 데이터 생성
    X, y = make_classification(n_samples=100, n_features=20, n_informative=15,
                               n_redundant=5, random_state=1)
    model = LogisticRegression() # 로지스틱 회귀 알고리즘 설정
    # 앞서 설정한 데이터와 회귀 알고리즘을 기반으로 한 score설정
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return mean(scores), scores.min(), scores.max() # 평균, 최소, 최대 점수 리턴

ideal, _, _ = evaluate_model(LeaveOneOut()) # 이상적인 테스트 상태 계산
print('Ideal: %.3f' % ideal)    # 상태 출력
folds = range(2, 20)    # 2부터 20까지 1씩 늘려 감
means, mins, maxs = list(), list(), list()  # 평균, 최소, 최대 리스트 만듦

for k in folds: # 각각의 k값마다 평균, 최소, 최대 값 계산 후 출력
    cv = KFold(n_splits=k, shuffle=True, random_state=1)
    k_mean, k_min, k_max = evaluate_model(cv)
    print('> folds=%d, accuracy=%.3f (%.3f,%.3f)' % (k, k_mean, k_min, k_max))
    # 평균 값이랑 오차 값 표출
    means.append(k_mean)
    mins.append(k_mean - k_min)
    maxs.append(k_max - k_mean)

# 데이터 편차 표시, 그래프 그리기, 그래프 표출
pyplot.errorbar(folds, means, yerr=[mins, maxs], fmt='o')   
pyplot.plot(folds, [ideal for _ in range(len(folds))], color='r')   
pyplot.show()
