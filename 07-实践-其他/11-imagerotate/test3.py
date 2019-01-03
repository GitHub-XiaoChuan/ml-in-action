import numpy as np

a = np.array([1,2,3,4,5,6,7,8,9,10])
b = np.where(a>5)
print(a)
print(b)
print(a[b])

start = np.percentile(a, 25)
end = np.percentile(a, 75)
print(start)
print(end)
b = np.array(a[b])
print(a[np.where(a>5)])
print(b[np.where(np.array(b<8))])
#c = np.where(np.where(a>5)<8)
# c = np.where([a>5, a<8])
#print(c)