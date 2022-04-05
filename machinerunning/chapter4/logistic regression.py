import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('User_Data.csv') # csv 파일 User_Data.csv 불러오기

x = dataset.iloc[:, [2, 3]].values  # 3,4번째 열 입력
y = dataset.iloc[:, 4].values  # 5번째 열 출력

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, random_state=0) # x, y의 train/test 분리

from sklearn.preprocessing import StandardScaler

# 메소드체이닝을 사용하여 fit과 transform을 연달아 호출 후 해당 fit으로 test데이터도 transform 해줌
sc_x = StandardScaler()
xtrain = sc_x.fit_transform(xtrain)
xtest = sc_x.transform(xtest)
print(xtrain[0:10, :])

from sklearn.linear_model import LogisticRegression

# 로지스틱 회귀 모델 생성 후 학습
classifier = LogisticRegression(random_state=0)
classifier.fit(xtrain, ytrain)
y_pred = classifier.predict(xtest)

from sklearn.metrics import confusion_matrix

# y데이터와 y예측데이터로 혼동행렬 표시
cm = confusion_matrix(ytest, y_pred)
print("혼동행렬 : \n", cm)

from sklearn.metrics import accuracy_score

print("정확도 : ", accuracy_score(ytest, y_pred)) # 혼동행렬에 사용된 데이터로 정확도 계산

from matplotlib.colors import ListedColormap

X_set, y_set = xtest, ytest # x데이터와 y데이터 설정
# 최댓값-1부터 최솟값+1까지 0.01간격으로 구성된 배열을 선언 후 직사각형 격자 생성
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1,
                               stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1,
                               stop=X_set[:, 1].max() + 1, step=0.01))

# x1, x2, x1과 x2 예측 재배열 한 것을 빨간색, 녹색으로 나뉘어진 등차선에 표시
plt.contourf(X1, X2, classifier.predict(
    np.array([X1.ravel(), X2.ravel()]).T).reshape(
    X1.shape), alpha=0.75, cmap=ListedColormap(('red', 'green')))

# x축과 y축의 한계 설정
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):    # 빨강, 녹색 두가지 인자 설정 후 표시
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)

# 등차선에 제목, x,y이름와 범례 추가 후 표시
plt.title('Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()