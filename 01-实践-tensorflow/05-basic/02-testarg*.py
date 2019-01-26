def f(*a):
    print(type(a))
    print(a)


f([1, 2, 3])
f((1, 2, 3))
f(1, 2, 3)
print(*[1, 2, 3])
print(*[1, 2, 3])
print(*(1, 2, 3))