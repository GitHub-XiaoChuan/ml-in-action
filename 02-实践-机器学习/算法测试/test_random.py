import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# random_rand = np.random.rand(100000) - 0.5
# rand1 = [round(x, 4) for x in random_rand]
# c = Counter(rand1)
# plt.scatter(c.keys(), c.values())
# plt.show()

normal_rand = np.random.normal(0, pow(100000, -0.5), 100000)
normal_rand = np.random.normal(0.0, 1.0, 100000)
rand2 = [round(x, 2) for x in normal_rand]
c2 = Counter(rand2)
plt.scatter(c2.keys(), c2.values())
plt.show()
