import matplotlib.pyplot as plt
import math

#x = range(-3, 3, 1)
#plt.plot(x, [math.exp(x1) for x1 in x])
#plt.show()

print(math.log(1))
print(math.log(math.e))
print(math.log(1/40)/30)
print(math.exp(- 0.1535056728662697 * 30)*100)

x = range(0, 100, 1)
plt.plot(x, [math.exp(- 0.1535056728662697 * x1) * 100 for x1 in x])
plt.show()