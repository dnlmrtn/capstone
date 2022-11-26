import random
import numpy as np
import scipy.integrate as solve_ivp

class patient:
    def __init__(self,state):
        self.state = state #state
        self.action = 0
        self.state_space = np.linspace(0,250,250/5)
        self.action_space = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] #possible doses

        self.target = 80 #target blood glucose level
        self.lower = 65 #below this level is dangerous, NO insulin should be administered
        self.upper = 105 #above this is dangerous, perceived optimal dose must be administered

    def dynamics(state, action, meal):
        g = state[0] #blood glucose (mg/dL)
        x = state[1] #remote insulin (micro-u/ml)
        i = state[2] #plasma insulin (micro-u/ml)
        q1 = state[3] #S1
        q2 = state[4] #S2
        g_gut = state[5] #gut blood glucose (mg/dL)

        
        #parameters (??)
        gb = 291.0     # (mg/dL)                    Basal Blood Glucose
        p1 = 3.17e-2   # 1/min
        p2 = 1.23e-2   # 1/min
        si = 2.9e-2    # 1/min * (mL/micro-U)
        ke = 9.0e-2    # 1/min                      Insulin elimination from plasma
        kabs = 1.2e-2  # 1/min                      t max,G inverse
        kemp = 1.8e-1  # 1/min                      t max,I inverse
        f = 8.00e-1    # L
        vi = 12.0      # L                          Insulin distribution volume
        vg = 12.0      # L                          Glucose distibution volume
        
        # Compute ydot:
        dydt = np.empty(6)

        dydt[0] = -p1 * (g - gb) - si * x * g + f * kabs / vg * g_gut + f / vg * meal  # (1)
        dydt[1] = p2 * (i - x)  # remote insulin compartment dynamics (2)
        dydt[2] = -ke * i + action  # plasma insulin concentration  (3)
        dydt[3] = action - kemp * q1  # two-part insulin absorption model dS1/dt
        dydt[4] = -kemp * (q2 - q1)  # two-part insulin absorption model dS2/dt
        dydt[5] = kemp * q2 - kabs * g_gut

        # convert from minutes to hours
        dydt = dydt * 60
        return dydt

    def sim_action(self, action):
        
        if self.state is None:
            raise Exception("Please reset() environment")
        
        self.dose = action

        time_step = [0, 10] #assume measurements are taken every 10 mins

        x_next = solve_ivp(self.dynamics, time_step, self.state[0:6], args = (action, self.state[6]))

        for i in range(10):
            self.state[i] = x_next[i][10]

        return self.state

    def reward(self):
        if self.state[0]<=self.lower:
            return 0
        if self.lower < self.state[0] < self.target:
            reward = (self.state[0] - self.lower)*10
            return reward
        if self.target < self.state[0] < self.upper:
            reward  = (self.upper - self.state[0])*5
            return reward
        if self.upper <= self.state[0]:
            return 0
        

# Updated Q Learning Function

# These dynamics and functions will be completed in the future, but I made blank ones because the QL
# will rely on them being gloabally defined

def qValUpdate(qtable, patient, action, alpha, gamma, w, lam):
    # find what the next state is going to be given the current state and action
    state2 = patient.sim_action(action)
    # find the action in the next state which gives highest q
    possible_Q = {}
    for a in patient.action_space:
        possible_Q.update({a, qtable[state2, a]})
    maxA = max(possible_Q, key = possible_Q.get)
    maxQ = possible_Q[maxA]

    # update using the Q learning equation
    qCurrent = qtable(patient.state,action)
    qNew = (1-alpha(patient.state,action))*qtable(patient.state, action) + alpha*(costFN(patient.state,action,w) + gamma*maxQ - qtable(patient.state,action))
    qtable[patient.state,action] = qNew

    qDif = qNew - qCurrent

    #given our next state, choose the action to take based on probability distribution
    if (patient.state[0] < patient.lower or patient.state[0] > patient.upper):
        action2 = maxA
    else:
        scores = qtable[state2]
        sumQ = sum(np.exp(-lam*scores))
        probs = np.exp(-lam*scores)/sumQ
        action2 = random.choices(scores,probs, k=1)

    return qtable, qDif, state2, action2



# Simulation

patient1 = patient([np.zeros(9)])

print(patient1)
