import numpy as np
from matplotlib import pyplot as plt

# 가우시안 표준 정규 분포에서 난수 matrix array 생성
X1 = np.random.randn(1000)
X2 = 10 + np.random.randn(1000)

# 생성한 데이터를 이용해 히스토그램 그린 후 출력 
plt.figure(figsize=(10,6))
plt.hist(X1, bins=20, alpha=0.4)
plt.hist(X2, bins=20, alpha=0.4)
plt.show()