from sklearn import datasets
import numpy as np

def test_split(index, value, dataset):  # 데이터 분할
    left, right = list(), list()
    for row in dataset: # 비교하는 값보다 작으면 왼쪽, 크면 오른쪽으로 분할
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

def gini_index(groups, classes):  # 지니지수 계산
    n_instances = float(sum([len(group) for group in groups])) # 분할하는 지점에서 모든 샘플들 카운트
    gini = 0.0  # 각 그룹들의 가중 지니 지수 합계 저장
    for group in groups:
        size = float(len(group))
        if size == 0:   # 0으로 나누는 경우 제외
            continue
        score = 0.0
        for class_val in classes: # 각 클래스의 점수에 따라 그룹의 점수 측정
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        gini += (1.0 - score) * (size / n_instances) # 상대적인 크기에 따라 그룹 점수 평가
    return gini

def get_split(dataset): # 데이터 집합에 가장 적합한 분할 지점 선택
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None # 변수 설정
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset) # 데이터 분할
            gini = gini_index(groups, class_values) # 지니지수 계산
            if gini < b_score: # 지니지수가 특정 점수 미만일 경우 대입
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}

def split(node, max_depth, min_size, depth):  # 분할
    left, right = node['groups']
    del (node['groups'])
    if not left or not right: # 분할되었는지 되지 않았는지 확인 
        node['left'] = node['right'] = to_terminal(left + right)
        return
    if depth >= max_depth:  # 최대 깊이를 넘었는지 확인 후 넘었으면 터미널 노드 값 생성
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    if len(left) <= min_size:   # 왼쪽 노드가 최솟값 이하면 터미널 노드 값 생성 그 외에 경우 분할
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth + 1)
    if len(right) <= min_size: # 오른쪽 노드가 최솟값 이하면 터미널 노드 값 생성 그 외에 경우 분할
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth + 1)


def to_terminal(group): # 터미널 노드 값 생성
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


def build_tree(train, max_depth, min_size):  # 결정트리 생성
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root


def print_tree(node, depth=0):  # 만든 결정 트리 출력
    if isinstance(node, dict):
        print('%s[X%d < %.3f]' % ((depth * '', (node['index'] + 1), node['value'])))
        print_tree(node['left'], depth + 1)
        print_tree(node['right'], depth + 1)
    else:
        print('%s[%s]' % ((depth * '', node)))

iris = datasets.load_iris() # 아이리스 데이터 불러오기
dataset = np.c_[iris.data, iris.target] # 두 데이터를 합침

tree = build_tree(dataset, 3, 1)  # 합친 데이터를 이용해 최대 깊이 3, 최소크기1인 결정트리 생성
print_tree(tree) # 결정트리 출력