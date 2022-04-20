import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

pdiabetes = pd.read_csv('diabetes.csv', header=None)    # 당뇨병 데이터 읽기
print(pdiabetes[0:5])

x = pdiabetes.iloc[1:,:8] # 2행부터 8열까지의 데이터
y = pdiabetes.iloc[1:,8:].values.flatten()  # 2행부터 8행까지의 데이터 플래튼
print('x shape: ', x.shape,' y shape: ', y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)   # x, y의 train/test 분리
std_scl = StandardScaler()  # Standardization 평균 0 / 분산 1
# 교차검증시
std_scl.fit(x_train)
x_train = std_scl.transform(x_train) #속성값 정규화
x_test = std_scl.transform(x_test)

svc = SVC(kernel='rbf') # rbf 커널 설정 후 학습
svc.fit(x_train, y_train)

print('학습 데이터 정확도 : ', svc.score(x_train, y_train)) # 학습 데이터 정확도 점수 계산후 출력
print('테스트 데이터 정확도 : ', svc.score(x_test, y_test))  # 테스트 데이터 정확도 점수 계산후 출력
