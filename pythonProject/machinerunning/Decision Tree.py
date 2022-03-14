import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree

iris = load_iris()  # 아이리스 데이터 불러오기
# 트리의 최대 깊이를 3으로 한다. random_state에 특정값을 지정해 함수 수행시 마다 동일한 트리가 만들어지게 함
# 엔트로피로 불순도 계산 방법을 설정한 뒤 아이리스 데이터 로딩하고 학습시켜 의사결정트리 만듦
decision_tree = DecisionTreeClassifier(criterion="entropy", random_state=0, max_depth=3)
decision_tree = decision_tree.fit(iris.data, iris.target)

# 만든 의사 결정 트리 출력
plt.figure()
plot_tree(decision_tree, filled=True)
plt.show()
