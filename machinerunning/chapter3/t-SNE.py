from sklearn import datasets
import pandas as pd
iris = datasets.load_iris() # 아이리스 데이터 불러오기

labels = pd.DataFrame(iris.target) # 아이리스 타겟 데이터 DataFrame 생성 후 열 설정
labels.columns=['labels']
data = pd.DataFrame(iris.data)  # 아이리스 데이터 DataFrame 생성 후 열 설정
data.columns=['Sepal length','Sepal width','Petal length','Petal width']
data = pd.concat([data, labels],axis=1) # 왼쪽 + 오른쪽으로 dataframe 합침
feature = data[['Sepal length','Sepal width','Petal length','Petal width']] # 데이터 설정

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

model = TSNE(learning_rate=100, perplexity=30)  # t-SNE를 이용하여 차원 감소 시킨 모델 생성 및 학습
transformed = model.fit_transform(feature)

# 학습한 모델을 이용하여 산점도 Xs,Ys,범례 설정 후 표시
xs = transformed[:,0]
ys = transformed[:,1]
plt.scatter(xs,ys,c=labels['labels'])

plt.show()