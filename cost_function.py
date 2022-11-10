import numpy as np

def costFN(xCurrent,uCurrent,uPast,W):
    return np.dot(xCurrent,W) + 0.155*uCurrent + 0.5*abs(uCurrent - uPast)







