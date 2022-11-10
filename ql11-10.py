#Q learning time
import numpy as np
# For basic computational reasons, assume anxiety only has one score from 0-5, instead of 14 scores.

# define possible actions
dose = [5,10,15,20,25]

# define our possible states
anx = [0,1,2,3,4]


# initialize Qlearning table


# function to update Q values
def qValUpdate(qtable, state, action, prevact, alpha,gamma,w):
    # find what the next state is going to be given the current state and action
    nextState = getNextStateFN(state,action)
    # find the action in the next state which gives highest q
    possible_states = []
    for a in dose:
        possible_states.append(qtable(nextState, a)) 
    maxQ = max(possible_states)
    # update using the Q learning equation
    qNew = qtable(state, action) + alpha*(-costFN(state,action,prevact,w) + gamma*maxQ - qtable(state,action))

    return qNew
# run the simulation
iters = 3       #set max iterations
qtable = np.zeros((len(dose),len(anx)), float)      #define empty qtable
for n in range(iters):
    for i in range(len(anx)):
        for j in range(len(dose)):
            qtable[i,j] = qValUpdate(qtable = )
            
#confusion: how can we incorporate the real data into this
# in this simulation, we need a way of storing and recalling the previous action for every agent
            
    

