#Q learning time
import numpy as np
def dynamics(xt, ut):
    xnext1 = (0.9)*xt-0.18*ut
    #xnext2 = (0.9)*xt[1]+0.07*ut
    #xnext.append(xnext1)
    #xnext.append(xnext2)
    return xnext1

#Uncertainty Function

from numpy.random import seed
from numpy.random import randint
'''
def uncertainty(xt):
    seed(1)
    randnum = randint(0,100,14)
    for i in range (0, len(xt)):
        #For Positive effects and decrease
        if i <= 7 and 3 <= randnum[i] <= 10 and xt[i] > 0:
            xt[i] -= 1
        #For Positive effects and increase    
        if i <= 7 and 0 <= randnum[i] <= 2 and xt[i] < 5:
            xt[i] += 1
        #For Negative effects and decrease
        if i > 7 and 3 <= randnum[i] <= 10 and xt[i] > 0:
            xt[i] -= 1
        #For Negative effects and increase    
        if i > 7 and 0 <= randnum[i] <= 2 and xt[i] < 5:
            xt[i] += 1
    return xt
'''
def costFN(xCurrent,uCurrent,W):
    return np.dot(xCurrent,W) + 0.155*uCurrent

# For basic computational reasons, assume anxiety only has one score from 0-5, instead of 14 scores.

# define possible actions
doses = [5,10,15,20,25]

# define our possible states
anx = [0,1,2,3,4]

# function to update Q values
def qValUpdate(qtable, state, action, alpha, gamma, w):
    # find what the next state is going to be given the current state and action
    nextState = dynamics(state,action)
    # find the action in the next state which gives highest q
    '''
    possible_states = {}
    for a in doses:
        possible_states.update({a, qtable[nextState, a]})
    maxA = max(possible_states, key = possible_states.get)
    maxQ = possible_states[maxA]
    '''
    possible_states = []
    for a in range(len(doses)):
        nextState = int(dynamics(state,action))
        #print(nextState)
        
        possible_states.append(qtable[nextState,a])
    maxQ = max(possible_states)
    #print(maxQ)
    # update using the Q learning equation
    qNew = qtable[state, action] + alpha*(-costFN(state,action,w) + gamma*maxQ - qtable[state,action])
    qtable[state,action] = qNew
    #print("new q is", qNew)
    return qNew
    '''
    if abs(qCurrent-qNew) >= 0.01:
        qValUpdate(qtable, dynamics(state,action),maxA, alpha,gamma,w)
    
    #
    return qNew
    '''

# run the simulation
iters = 3       #set max iterations
qtable = np.zeros((len(doses),len(anx)), float)      #define empty qtable
qValUpdate(qtable,4,1, 0.1,0.1,0.1)

for n in range(iters):
    for i in range(len(anx)):
        for j in range(len(doses)):
            qtable[i,j] = qValUpdate(qtable,i,j, 0.1,0.1,0.1)
print(qtable)

#confusion: how can we incorporate the real data into this
# in this simulation, we need a way of storing and recalling the previous action for every agent