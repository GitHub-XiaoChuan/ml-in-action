import matplotlib.pyplot as plt
import numpy as np

# 初始化图形
m = 600
n = 400
mn = np.zeros((m, n))

for i in range(m):
    mn[i, :] = 255*(i/m)

plt.imshow(mn)
plt.show()

#

