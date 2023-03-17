import numpy as np

q = np.zeros((3,3,2))

q[0,0][0]=1
q[0,0][1]=10

print(q[0,0])
print(q[0,0][1])
