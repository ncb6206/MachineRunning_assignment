from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 표본 데이터수 1000, 종속 변수의 클래스 수 2, 독립 변수의 수 20, 랜덤 시드 설정한 가상의 분류 모형 데이터 생성
X, y = make_classification(n_samples=1000, n_classes=2, n_features=20, random_state=27)
# 데이터 X 와 레이블 y 설정
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=27)

from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression()   # 로지스틱 회귀 알고리즘 설정
model1.fit(X_train, y_train)    # 모델에 데이터와 레이블 fit
pred_prob1 = model1.predict_proba(X_test)   # 모델 데이터의 개별 예측 확률

from sklearn.metrics import roc_curve
fpr1, tpr1, thresh1 = roc_curve(y_test, pred_prob1[:,1], pos_label=1) # 레이블과 예측 확률 계산 후 ROC 곡선 그리기

random_probs = [0 for i in range(len(y_test))]  # 0부터 레이블의 크기 만큼 랜덤으로 설정
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1) # 레이블과 랜덤 예측 확률 계산 후 ROC 곡선 그리기

from sklearn.metrics import roc_auc_score
# AUC에서 예측 점수로부터 계산 후 출력
auc_score1 = roc_auc_score(y_test, pred_prob1[:,1])
print('AUC value : %s' % auc_score1)    

import matplotlib.pyplot as plt
# 각각의 곡선 색깔과 가로와 세로 단위, 타이틀 설정
plt.style.use('seaborn')
plt.plot(fpr1, tpr1, linestyle='--', color='orange', label='Logistic Regression')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')

# ROC 곡선 표시
plt.legend(loc='best')
plt.savefig('ROC', dpi=300)
plt.show();
