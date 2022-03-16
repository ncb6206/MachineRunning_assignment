# multi-class classification
from sklearn.datasets import make_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn. linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# 표본 데이터수 1000, 종속 변수의 클래스 수 3, 독립 변수의 수 20, 독립 변수 중 종속 변수와 상관 관계가 있는 성분의 수 3, 랜덤 시드 설정한 가상의 분류 모형 데이터 생성
X, y = make_classification(n_samples=1000, n_classes=3, n_features=20, n_informative=3, random_state=42)
# 데이터 X 와 레이블 y 설정
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
clf = OneVsRestClassifier(LogisticRegression()) # 타깃 레이블 개수만큼 로지스틱 회귀 모델 생성
clf.fit(X_train, y_train)   # 모델에 데이터와 레이블 fit
pred = clf.predict(X_test)  # 모델 데이터의 전체적인 예측 확률
pred_prob = clf.predict_proba(X_test) # 모델 데이터의 개별 예측 확률

# 변수 초기화
fpr = {}
tpr = {}
thresh ={ }
n_class = 3
# 각각의 클래스 별로 레이블과 예측 확률 계산 후 ROC 곡선 그리기
for i in range(n_class):
    fpr[i], tpr[i], thresh[i] = roc_curve(y_test, pred_prob[:,i], pos_label=i)

# 각각의 곡선 색깔과 가로와 세로 단위, 타이틀 설정 후 곡선 표시
plt.plot(fpr[0], tpr[0], linestyle='--', color='orange', label='Class O vs Rest')
plt.plot(fpr[1], tpr[1], linestyle='--', color='green', label='Class 1 vs Rest')
plt.plot(fpr [2], tpr[2], linestyle='--', color='blue', label='Class 2 vs Rest')
plt.title('Multiclass ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.savefig('Multiclass ROC', dpi=300)
plt.show();

