import collections
import numpy as np

box = np.array([
    [1, 1, 2, 2, 0.3],
    [1, 1, 3, 3, 0.4],
    [1, 1, 4, 4, 0.5],
    [2, 2, 3, 3, 0.6],
    [2, 2, 4, 4, 0.7],
])
for b in box:
    print(b)



print(box.flatten())

print(map(float, box.flatten()))

print(collections.OrderedDict(zip(['x0', 'y0', 'x1', 'y1', 'score'], map(float, box.flatten()))))

#collections.OrderedDict(zip(['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3'], map(float, box.flatten())))