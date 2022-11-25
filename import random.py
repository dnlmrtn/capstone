import random
import numpy as np
numbers = [num for num in range(5)]
print(numbers)
val = [0.9,0.025,0.025,0.025,0.025]

x = random.choices(numbers,weights = val, k=1)
print(x)
qtable = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(qtable[0])

print(np.exp(qtable[0]))

