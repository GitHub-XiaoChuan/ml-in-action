import sys
import copy

class Point:
    __slots__ = ("x", "y")

    def __init__(self, x, y):
        print(x, y)
        self.x = x
        self.y = y

def make_object(Class, *args, **kwargs):
    return Class(*args, **kwargs)

# 第一种
point1 = Point(1, 2)

# 第二种
point2 = eval("{}({}, {})".format("Point", 2, 4))

# 第三种
point3 = getattr(sys.modules[__name__], "Point")(3, 6)

# 第四种
point4 = globals()["Point"](4, 8)

# 第五种
point5 = make_object(Point, 5, 10)

# 第六种
point6 = copy.deepcopy(point5)
point6.x = 6
point6.y = 12

# 第七种
point7 = point1.__class__(7, 14)