#Q learning time
import numpy as np
def dynamics(xt = list, ut = list):
    xnext = []
    xnext1 = (0.9)*xt-0.18*ut
    #xnext2 = (0.9)*xt[1]+0.07*ut
    xnext.append(xnext1)
    #xnext.append(xnext2)
    return xnext

def costFN(xCurrent,uCurrent,uPast,W):
    return np.dot(xCurrent,W) + 0.155*uCurrent

# For basic computational reasons, assume anxiety only has one score from 0-5, instead of 14 scores.

# define possible actions
dose = [5,10,15,20,25]

# define our possible states
anx = [0,1,2,3,4]
test = {'a': 1, 'b': 3000, 'c': 0}

value = max(test, key=test.get)
print(value, test[value])

# initialize Qlearning table

# function to update Q values
def qValUpdate(qtable, state, action, alpha, gamma, w):
    # find what the next state is going to be given the current state and action
    nextState = dynamics(state,action)
    # find the action in the next state which gives highest q
    possible_states = {}
    for a in dose:
        possible_states.update({a, qtable[nextState, a]})
    maxA = max(possible_states, key = possible_states.get)
    maxQ = possible_states[maxA]
    # update using the Q learning equation
    qCurrent = qtable(state,action)
    qNew = qtable(state, action) + alpha*(-costFN(state,action,w) + gamma*maxQ - qtable(state,action))
    qtable[state,action] = qNew
    if abs(qCurrent-qNew) >= 0.01:
        qValUpdate(qtable, dynamics(state,action),maxA, alpha,gamma,w)

    #
    return qNew
# run the simulation
iters = 3       #set max iterations
qtable = np.zeros((len(dose),len(anx)), float)      #define empty qtable
for n in range(iters):
    for i in range(len(anx)):
        for j in range(len(dose)):
            qtable[i,j] = qValUpdate(qtable,i,j, 0.1,0.1,0.1)
print(qtable)
#confusion: how can we incorporate the real data into this
# in this simulation, we need a way of storing and recalling the previous action for every agent