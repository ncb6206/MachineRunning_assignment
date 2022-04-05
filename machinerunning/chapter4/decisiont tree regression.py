from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('housing.data', header=None, sep='\s+') # csv 파일 housing.data 불러오기
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', "AGE", 'DIS', 'RAD',
        'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'] # 칼럼명(필드) 목록 설정 및 출력
print(df.head())
X = df[['LSTAT']].values
y = df['MEDV'].values

tree = DecisionTreeRegressor(max_depth=3)  # 최대 깊이 3인 결정 트리 회귀 모델 설정
tree.fit(X,y)

sort_idx = X.flatten().argsort() # 작은값부터 순서대로 X 데이터의 위치를 반환
# 산점도 X1, X2, 색깔 설정 후 표시
plt.scatter(X[sort_idx], y[sort_idx], c='lightblue')
# X값과 X예측값, 색깔, 선 두께 설정 후 표시 
plt.plot(X[sort_idx], tree.predict(X[sort_idx]), color='red', linewidth=2)
plt.xlabel('LSTAT(% Lower Status of the Population)')
plt.ylabel('MEDV(Price in $1000)')
plt.show()
