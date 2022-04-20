import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

digits = datasets.load_digits() # 글씨로 표현된 숫자 데이터 불러오기
_, axes = plt.subplots(2, 5)    # figure=2, axdes=5 받고 여러 그래프 보여줌
images_and_labels = list(zip(digits.images, digits.target))     # 숫자 이미지와 일치하는 타겟 리스트 설정
for ax, (image, label) in zip(axes[0, :], images_and_labels[:5]):
    ax.set_axis_off()   # x와 y축을 끔
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')   # 받은 이미지를 보여줌
    ax.set_title('Training: %i' % label)    # 각 이미지의 라벨 표시

n_samples = len(digits.images)  # 숫자 이미지 데이터의 갯수
data = digits.images.reshape((n_samples, -1))   # 숫자 이미지 데이터 재배열

X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.5, shuffle=False) # x, y의 train/test 분리
classifier = svm.SVC(kernel='rbf', gamma=0.001) # rbf 커널, 감마값 0.001 설정 후 학습
classifier.fit(X_train, y_train)
predicted = classifier.predict(X_test)  # xtest를 이용해 예측값 설정

test_data = X_test.reshape((len(X_test), 8, 8)) # xtest를 재배열한 테스트 데이터 생성
images_and_predictions = list(zip(test_data, predicted))    # 테스트 데이터와 예측값 리스트 설정
for ax, (image, prediction) in zip(axes[1, :], images_and_predictions[:5]):
    ax.set_axis_off()   # x와 y축을 끔
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')   # 받은 이미지를 보여줌
    ax.set_title('Predict: %i' % prediction)    # 예측값 표시

print("SVM 분류 결과 %s:\n%s\n"
      % (classifier, metrics.classification_report(y_test, predicted))) # ytest와 예측값을 정의
disp = metrics.plot_confusion_matrix(classifier, X_test, y_test) # 혼동행렬의 그림을 출력
disp.figure_.suptitle("Confusion Matrix")   # 혼동행렬 타이틀 출력
print("혼동 행렬: \n%s" % disp.confusion_matrix)    # 혼동행렬 출력
print("정확도 : ", accuracy_score(y_test, predicted))  # ytest와 예측값을 계산하여 정확도 출력

plt.show()