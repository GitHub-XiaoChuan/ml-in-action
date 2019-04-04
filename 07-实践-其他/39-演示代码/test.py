lr = 0.0
ya = 0
yu = 0
for i in range(401):
    for j in range(301):
        if (i*100+j*125) <= 50000:
            lrt = i*5 + j*4
            if lr < lrt:
                lr = lrt
                ya = i
                yu = j
print(ya)
print(yu)
print(lr)