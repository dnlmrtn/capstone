import random
import numpy as np
import scipy.integrate as integrate
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class patient:
    def __init__(self,state):
        self.t = 0
        self.state = state #state
        self.action = 0
        self.state_space = np.linspace(0, 250, 50)
        self.action_space = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10) #possible doses

        self.target = 80 #target blood glucose level
        self.lower = 65 #below this level is dangerous, NO insulin should be administered
        self.upper = 105 #above this is dangerous, perceived optimal dose must be administered
        self.hash = {}
        i = 0
        for s in self.state_space:
            self.hash.update({s : i})
            i += 1
        self.meals = [1259,1451,1632,1632,1468,1314,1240,1187,1139,1116,
                  1099,1085,1077,1071,1066,1061,1057,1053,1046,1040,
                  1034,1025,1018,1010,1000,993,985,976,970,964,958,
                  954,952,950,950,951,1214,1410,1556,1603,1445,1331,
                  1226,1173,1136,1104,1088,1078,1070,1066,1063,1061,
                  1059,1056,1052,1048,1044,1037,1030,1024,1014,1007,
                  999,989,982,975,967,962,957,953,951,950,1210,1403,
                  1588,1593,1434,1287,1212,1159,1112,1090,1075,1064,
                  1059,1057,1056,1056,1056,1055,1054,1052,1049,1045,
                  1041,1033,1027,1020,1011,1003,996,986]

    def dynamics(self, t, y, ui, d):
        g = y[0]                # blood glucose (mg/dL)
        x = y[1]                # remote insulin (micro-u/ml)
        i = y[2]                # insulin (micro-u/ml)
        q1 = y[3]
        q2 = y[4]
        g_gut = y[5]            # gut blood glucose (mg/dl)

            # Parameters:
        gb    = 291.0           # Basal Blood Glucose (mg/dL)
        p1    = 3.17e-2         # 1/min
        p2    = 1.23e-2         # 1/min
        si    = 2.9e-2          # 1/min * (mL/micro-U)
        ke    = 9.0e-2          # 1/min
        kabs  = 1.2e-2          # 1/min
        kemp  = 1.8e-1          # 1/min
        f     = 8.00e-1         # L
        vi    = 12.0            # L
        vg    = 12.0            # L

            # Compute ydot:
        dydt = np.empty(6)
        dydt[0] = -p1*(g-gb) - si*x*g + f*kabs/vg * g_gut + f/vg * d
        dydt[1] =  p2*(i-x) # remote insulin compartment dynamics
        dydt[2] = -ke*i + ui # insulin dynamics
        dydt[3] = ui - kemp * q1
        dydt[4] = -kemp*(q2-q1)
        dydt[5] = kemp*q2 - kabs*g_gut

            # convert from minutes to hours
        dydt = dydt*60
        return dydt

       

    def sim_action(self, action):
        
        #print(self.state)
        if self.state is None:
            raise Exception("Please reset() environment")
        
        self.t = (self.t + 1) % 102
        self.state[7] = self.state[0]
        self.state[8] = self.state[6]

        time_step = np.array([0,10]) #assume measurements are taken every 10 mins
        y0 = np.array(self.state[0:6])
        meal = self.state[6]

        x = solve_ivp(self.dynamics, time_step, y0, args = (action, meal))
        #print('sim done, action was:')
        #print(action)
        for i in range(6):
            self.state[i] = x.y[i][-1]
        
        self.state[6] = self.meals[self.t]

        #print(self.state)
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
# will rely on them being globally defined

def qValUpdate(qtable, patient, action, alpha, gamma, lam):
    # find what the next state is going to be given the current state and action
    q_state1 = patient.state[0]
    closest = 0
    for s in patient.state_space:
        if abs(q_state1 - s) < abs(q_state1 - closest):
            closest = s
    q_state1 = patient.hash[closest]

    state2 = patient.sim_action(action)

    q_state2 = state2[0]
    closest = 0
    for s in patient.state_space:
        if abs(q_state2 - s) < abs(q_state2 - closest):
            closest = s
    q_state2 = patient.hash[closest]
    # find the action in the next state which gives highest q
    possible_Q = {}
    for a in patient.action_space:
        possible_Q.update({a : qtable[q_state2, a]})
    maxA = max(possible_Q, key = possible_Q.get)
    maxQ = possible_Q[maxA]

    # update using the Q learning equation
    qCurrent = qtable[q_state1,action]
    qNew = (1-alpha)*qCurrent + alpha*(patient.reward() + gamma*maxQ - qCurrent)
    qtable[q_state1,action] = qNew

    qDif = qNew - qCurrent

    #given our next state, choose the action to take based on probability distribution
   # if (patient.state[0] < patient.lower or patient.state[0] > patient.upper):
        #action2 = maxA
    #else:
   #scores = qtable[q_state2]
    #sumQ = sum(np.exp(-lam*scores))
    #probs = np.exp(-lam*scores)/sumQ
    #action2 = random.choices(scores,probs, k=1)
    action2 = random.choice(patient.action_space)
    #action2 = action
    return qtable, qDif, state2, action2


#Meals


# Simulation

patient1 = patient(np.zeros(9))
patient1.state[0] = 80
patient1.state[1] = 30
patient1.state[2] = 30
patient1.state[3] = 17
patient1.state[4] = 17
patient1.state[5] = 250
patient1.state[6] = 1000

t = 0

Q = np.zeros((len(patient1.state_space), len(patient1.action_space)))
action = 10
while t <= 1000:
    t += 1
    Q, qDif, patient1.state, action = qValUpdate(Q, patient1, action, 0.1, 0.1, 0.1)

print(Q)


#Plot Creation

x1 = [0]*(len(Q))
y1 = [0]*(len(Q))

for i in range(0, len(x1)):
    x1[i] = 5*i
    Qdose = 0
    for j in range(0, len(Q[0])):
        if (Q[i][j] > Qdose):
            Qdose = Q[i][j]
            y1[i] = j
print('x1', x1)
print('y1', y1)
plt.plot(x1, y1)

plt.xlabel('Blood Glucose Level')
plt.ylabel('Insulin Dosage')

plt.title('Q-Learning Dosage Recomendations')

plt.show()

