import numpy as np


print("Hello World")
arr = np.arange(9).reshape((3, 3))
print(arr)
print()
np.random.shuffle(arr)
print(arr)