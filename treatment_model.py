#Heres a basic model that Serdar was recommending

def dynamics(xt, ut):
    xnext = []
    xnext1 = (0.9)*xt[0]-0.18*ut
    xnext2 = (0.9)*xt[1]+0.07*ut
    xnext.append(xnext1)
    xnext.append(xnext2)
    return xnext

#Uncertainty Function
import numpy
from numpy.random import seed
from numpy.random import randint

def uncertainty(xt):
    seed(1)
    randnum = randint(0,100,14)
    for i in range (0, len(xt)):
        #For Positive effects and decrease
        if i <= 7 and 3 <= randnum[i] <= 10:
            xt[i] -= 1
        #For Positive effects and increase    
        if i <= 7 and 0 <= randnum[i] <= 2:
            xt[i] += 1
        #For Negative effects and decrease
        if i > 7 and 3 <= randnum[i] <= 10:
            xt[i] -= 1
        #For Negative effects and increase    
        if i > 7 and 0 <= randnum[i] <= 2:
            xt[i] += 1
    return xt
