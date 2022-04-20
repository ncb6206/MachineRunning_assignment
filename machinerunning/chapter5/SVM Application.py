from cvxopt import matrix, solvers, mul, spmatrix
import numpy as np

x = np.array([[1.,6.], [1.,8.],[4.,11.],[5.,2.],[7.,6.],[9.,3.]])   # 6차원 배열 x 설정
xt = np.transpose(x)    # 배열의 축 반전
XXt = np.dot(x,xt)  # 배열간의 내적 연산 수행
y = np.array([[1.],[1.],[1.],[-1.],[-1.],[-1.]])    # 6차원 배열 y 설정
yt = np.transpose(y)    # 배열의 축 반전
yyt = np.dot(y,yt)  # 배열간의 내적 연산 수행
H = np.multiply(XXt, yyt)   #XXt와 yyt의 곱셈 연산 수행 후 행렬 생성
H = matrix(H)

f = matrix([-1., -1., -1., -1., -1., -1],(6,1), 'd')    # 행렬 생성
A = np.diag([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0])   # 대각 행렬 생성 후 행렬 생성
A = matrix(A)
a = matrix([0., 0., 0., 0., 0., 0.],(6,1), 'd') # 행렬 생성
B = matrix([1, 1, 1, -1, -1, -1], (1,6),'d')    # 행렬 생성
b = matrix(0.0, (1,1), 'd') # 행렬 생성

sol = solvers.qp(H, f, A, a, B, b) # 수식을 이용하여 목표값 계산
print('\n', 'alpha_1 = ',sol['x'][0])
print(' alpha_2 = ', sol['x'][1])
print(' alpha_3 = ', sol['x'][2])
print(' alpha_4 = ', sol['x'][3])
print(' alpha_5 = ', sol['x'][4])
print(' alpha_6 = ', sol['x'][5])
