import numpy as np

q = np.zeros((2,2,2))
q[0,0,0]=1
q[1,0,0]=2
index = [0,1,2]
coords = (1,0)
coords = coords + (2,)*3

for i in range(3):
    i = int(i)
    coords = coords + (i,)

print(coords)


