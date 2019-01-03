import numpy as np

a = np.array([[1, 2, 1],[2, 1, 1], [1, 1, 2], [2, 2, 2], [2, 1, 2]])
b = np.array([2, 2, 2])
c = np.linalg.norm(a - b, axis=1)
print(c)