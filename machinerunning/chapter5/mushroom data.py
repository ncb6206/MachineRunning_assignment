import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

mushroom = pd.read_csv('mushrooms.csv', header=None)    # 버섯 데이터 가져오기
print(mushroom.head(4))

# 속성, 부류 배열 설정
X=[]
Y=[]
for idx,row in mushroom[1:].iterrows():
    Y.append(row.loc[0])    # 부류에 버섯 데이터 1행에 있는 데이터 삽입
    row_x=[]
    for v in row.loc[1:]:
        row_x.append(ord(v))    # 버섯 데이터에 있는 문자에 해당하는 유니코드 정수를 반환 후 삽입
    X.append(row_x) # 속성에 데이터 삽입

# 1~3행까지 속성, 부류 배열 출력
print('\n속성: \n', X[0:3])
print('\n부류: \n', Y[0:3])
x_train, x_test, y_train, y_test=train_test_split(X,Y,test_size=0.25)   # x, y의 train/test 분리

svc=SVC()   # 커널 설정 후 학습
svc.fit(x_train,y_train)

print('학습 데이터 정확도 : ', svc.score(x_train,y_train))  # 학습 데이터 정확도 점수 계산후 출력
print('테스트 데이터 정확도 : ', svc.score(x_test,y_test))   # 테스트 데이터 정확도 점수 계산후 출력