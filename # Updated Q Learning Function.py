# Updated Q Learning Function
#Adding in a probabilistic choice method
import random
import numpy as np
# These dynamics and functions will be completed in the future, but I made blank ones because the QL
# will rely on them being gloabally defined
def dynamics(xt = list, ut = list):
    xnext = 0
    return xnext

def costFN(xcurrent,uCureent):
    xcurrent - [1,2]
    return 0

dose = [5,10,15,20,25]

# define our possible states
gluc = [0,1,2,3,4]

def qValUpdate(qtable, state_space, action_space, state, action, alpha, gamma, w, lam):
    # find what the next state is going to be given the current state and action
    state2 = sim_action(state,action)
    # find the action in the next state which gives highest q
    possible_states = {}
    for a in action_space:
        possible_states.update({a, qtable[state2, a]})
    maxA = max(possible_states, key = possible_states.get)
    maxQ = possible_states[maxA]

    # update using the Q learning equation
    qCurrent = qtable(state,action)
    qNew = (1-alpha(state,action))*qtable(state, action) + alpha*(costFN(state,action,w) + gamma*maxQ - qtable(state,action))
    qtable[state,action] = qNew

    qDif = qNew - qCurrent

    #given our next state, choose the action to take based on probability distribution
    if (state[0] < patient.lower or state[0] > patient.upper):
        action2 = maxA
    else:
        scores = qtable[state2]
        sumQ = sum(np.exp(-lam*scores))
        probs = np.exp(-lam*scores)/sumQ
        action2 = random.choices(scores,probs, k=1)

    return qtable, qDif, state2, action2
