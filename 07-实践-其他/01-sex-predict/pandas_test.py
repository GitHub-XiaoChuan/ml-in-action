import pandas as pd
import numpy as np

data_3_4 = pd.DataFrame(np.arange(10,22).reshape(3,4))
print("原始")
print(data_3_4)
print("----")
print(data_3_4[:][np.arange(0,3)])
print("----")
print(data_3_4[:][3])

print(np.logspace(-4, 4, 3))