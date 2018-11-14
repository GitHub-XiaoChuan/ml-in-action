import numpy as np
import matplotlib.pyplot as plt

a = 0.0001

# 测试直线
x = np.linspace(-10, 10, 100)

y1 = x + 1
plt.subplot(3,2,1)
plt.plot(x, y1)
plt.title('y=x+1')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-10, 10)
plt.ylim(-10, 10)

y2 = x + a*x*x + 1
plt.subplot(3,2,2)
plt.plot(x, y2)
plt.title('y=x+a*x^2+1')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-10, 10)
plt.ylim(-10, 10)

y3 = x + a*x*x + a*x*x*x + 1
plt.subplot(3, 2, 3)
plt.plot(x, y3)
plt.title('y=x+a*x+a*x^2+a*x^3+1')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-10, 10)
plt.ylim(-10, 10)

y4 = x + a*x*x + a*x*x*x + a*x*x*x*x + 1
plt.subplot(3, 2, 4)
plt.plot(x, y4)
plt.title('y=x+a*x+a*x^2+a*x^3+a*x^4+1')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-10, 10)
plt.ylim(-10, 10)

y5 = x + a*x*x + a*x*x*x + a*x*x*x*x + a*x*x*x*x*x + 1
plt.subplot(3, 2, 5)
plt.plot(x, y5)
plt.title('y=x+a*x+a*x^2+a*x^3+a*x^4+a*x^5+1')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-10, 10)
plt.ylim(-10, 10)

y6 = x + a*x*x + a*x*x*x + a*x*x*x*x + a*x*x*x*x*x + a*x*x*x*x*x*x + 1
plt.subplot(3, 2, 6)
plt.plot(x, y6)
plt.title('y=x+a*x+a*x^2+a*x^3+a*x^4+a*x^5+a*x^6+1')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-10, 10)
plt.ylim(-10, 10)

plt.show()