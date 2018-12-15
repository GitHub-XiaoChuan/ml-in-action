import numpy as np
import matplotlib.pyplot as plt
import math


def sigma(x):
    return 1 / (1 + math.exp(-x))


a = np.arange(0, 1, .01)

plt.plot(a, [sigma(x * 100 - 34) for x in a])
plt.show()

plt.plot(a, [sigma(x * 999 - 400) for x in a])
plt.show()

plt.plot(a, [sigma(sigma(a1 * 999 - 400) * 1.0 + sigma(a2 * 100 - 20) * 0.8 + 3) for a1, a2 in zip(a, a)])
plt.show()

plt.plot(a, [sigma(sigma(a1 * 999 - 400) * 0.8 - sigma(a2 * 100 - 20) * 0.8 + 3) for a1, a2 in zip(a, a)])
plt.show()
