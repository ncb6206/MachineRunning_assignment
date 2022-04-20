from sklearn.datasets import fetch_lfw_people
from sklearn.svm import SVC
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

faces = fetch_lfw_people(min_faces_per_person=60)   # 실제 얼굴에 대한 이미지 데이터 60개 불러옴
print(faces.target_names)   # 이름 출력
print(faces.images.shape)   # 얼굴 이미지 출력
Xtrain, Xtest, ytrain, ytest = train_test_split(faces.data, faces.target, random_state=0)   # x, y의 train/test 분리

fig, ax = plt.subplots(3,5) # figure=3, axdes=5 받고 여러 그래프 보여줌
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap='bone')    # 얼굴 이미지 보여줌
    axi.set(xticks=[], yticks=[],
            xlabel=faces.target_names[faces.target[i]]) # 얼굴 이미지와 일치하는 이름 출력

pca = RandomizedPCA(n_components=150, whiten=True, random_state=0)  # PCA모델 학습
svc = SVC(kernel='rbf', class_weight='balanced')    # rbf 커널 설정
model = make_pipeline(pca, svc) # PCA모델과 rbf 커널을 이용해 파이프라인 만듦

param_grid = {'svc__C':[1,5,10,50],'svc__gamma':[0.0001,0.0005,0.001,0.005]}    # C와 감마값 설정
grid = GridSearchCV(model, param_grid)  # 모델과 파라미터를 이용하셔 가장 최적의 파라미터 찾은 후 학습
grid.fit(Xtrain,ytrain)
print(grid.best_params_)
model = grid.best_estimator_    # 최적의 파라미터로 학습된 모델 설정
yfit = model.predict(Xtest)     # 모델의 예측값 설정


fig, ax = plt.subplots(4,6) # figure=4, axdes=6 받고 여러 그래프 보여줌
for i, axi in enumerate(ax.flat):
    axi.imshow(Xtest[i].reshape(62,47), cmap='bone')    # xtest의 (62,47)를 2차원으로 바꾼 이미지 표시
    axi.set(xticks=[],yticks=[])
    axi.set_ylabel(faces.target_names[yfit[i]].split()[-1],
                   color='black' if yfit[i] == ytest[i] else 'red') # 타겟의 이름과 얼굴이 일치하면 검은색 일치하지 않으면 빨간색 표시
fig.suptitle('Predicted Names(Incorrect Labels in Red)', size=14);  # 보조 타이틀 설정
print(classification_report(ytest,yfit,target_names=faces.target_names))    # 타겟의 얼굴과 이름 표시
plt.show()

mat = confusion_matrix(ytest, yfit) # ytest와 yfit를 이용하여 혼동 행렬 생성
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=faces.target_names, yticklabels=faces.target_names) # 획득한 혼동 행렬을 히트맵으로 표현
# x축과 y축의 라벨을 설정한 후 표시
plt.xlabel('true label')
plt.ylabel('predicted label');
plt.show()

from sklearn.metrics import accuracy_score
print('정확도: ', accuracy_score(ytest,yfit))  # ytest와 yfit를 이용하여 정확도 계산 후 표시