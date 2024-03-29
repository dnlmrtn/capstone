import random
import numpy as np
import scipy.integrate as integrate
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
# #import seaborn as sn

def mealdynamics(a, b, t):
            
            meal = a * b ** t

            return meal

class patient:
    def __init__(self,state):
        self.t = 0
        self.last = -1
        self.state = state #state
        self.actions = []
        self.glucose = []
        self.time = []
        self.state_space = np.linspace(0, 250, 30)
        self.action_space = np.linspace(2, 6, 50) #possible doses
        self.total_reward = 0

        self.meal_space = np.linspace(800, 2000, 10) 

        self.target = 80 #target blood glucose level
        self.lower = 65 #below this level is dangerous, NO insulin should be administered
        self.upper = 105 #above this is dangerous, perceived optimal dose must be administered
        self.hash = {}
        i = 0
        self.b = 10
        self.l = 80
        self.d = 150

        self.meals = [1259,1355,1451,1542,1632,1632,1632,1550,1468,1391,1314,1277,1240,1214,1187,1163,1139,1128,1116,
                  1108,1099,1092,1085,1081,1077,1074,1071,1069,1066,1064,1061,1059,1057,1055,1053,1050,1046,1043,1040,
                  1037,1034,1030,1025,1022,1018,1014,1010,1005,1000,997,993,989,985,981,976,973,970,967,964,961,958,
                  956,954,953,952,951,950,950,950,951,951,1083,1214,1312,1410,1483,1556,1580,1603,1524,1445,1388,1331,
                  1279,1226,1200,1173,1155,1136,1120,1104,1096,1088,1083,1078,1074,1070,1068,1066,1065,1063,1062,1061,
                  1060,1059,1058,1056,1054,1052,1050,1048,1046,1044,1041,1037,1034,1030,1027,1024,1019,1014,1011,1007,
                  1003,999,994,989,986,982,979,975,971,967,965,962,960,957,955,953,952,951,951,950,1080,1210,1307,1403,
                  1496,1588,1591,1593,1514,1434,1361,1287,1250,1212,1186,1159,1136,1112,1101,1090,1083,1075,1070,1064,
                  1062,1059,1058,1057,1057,1056,1056,1056,1056,1056,1055,1055,1054,1054,1053,1052,1051,1049,1047,1045,
                  1043,1041,1037,1033,1030,1027,1024,1020,1016,1011,1007,1003,1000,996,991,986]


        self.hash.update({(-1) : i})    
        for last in range(len(self.meals)):
            self.hash.update({(last) : i}) #i indicates the time since last meal
            i += 1

        #self.meals = [1259,1451,1632,1632,1468,1314,1240,1187,1139,1116,
        #          1099,1085,1077,1071,1066,1061,1057,1053,1046,1040,
        #          1034,1025,1018,1010,1000,993,985,976,970,964,958,
        #          954,952,950,950,951,1214,1410,1556,1603,1445,1331,
        #          1226,1173,1136,1104,1088,1078,1070,1066,1063,1061,
        #          1059,1056,1052,1048,1044,1037,1030,1024,1014,1007,
        #          999,989,982,975,967,962,957,953,951,950,1210,1403,
        #          1588,1593,1434,1287,1212,1159,1112,1090,1075,1064,
        #          1059,1057,1056,1056,1056,1055,1054,1052,1049,1045,
        #          1041,1033,1027,1020,1011,1003,996,986]

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
       # dydt = dydt*60
        return dydt*60
    

    def sim_action(self, action):
        
        self.state[9] = self.state[10] #previous action
        self.state[10] = action #current action
        self.actions.append(action)         # log the action
        self.glucose.append(self.state[0])  # log the reading
        #print(self.state)
        if self.state is None:
            raise Exception("Please reset() environment")
        if self.t % len(self.meals) == 0:


            #Randomly generate Breakfast -> Dinner times (When carbpydrate levels will begin to rise)
            self.b = Btime = random.randint(5, 17)
            self.l = Ltime = random.randint(72, 84)
            self.d = Dtime = random.randint(146, 158)

            #Randomly generate peak carbohydrate amounts (occurs approximately 30-50min after initial consumption)
            Bcarb = random.randint(1400, 1625)
            Lcarb = random.randint(1450, 1650)
            Dcarb = random.randint(1475, 1675)

            #Time to digest
            tDigest = 6

            for t in range(len(self.meals)):

                if t == 0:
                    temptime = 0
                    a = 958
                    b = (950/a)**(1/(62))

                elif t == Btime:
                    temptime = 0
                    a = self.meals[t-1]
                    b = (Bcarb/a)**(1/(tDigest))

                elif (t == Btime + tDigest):
                    temptime = 0
                    a = self.meals[t-1]
                    b = (950/a)**(1/(62))

                elif ( t == Ltime):
                    temptime = 0
                    a = self.meals[t-1]
                    b = (Lcarb/a)**(1/(tDigest))

                elif (t == Ltime + tDigest):
                    temptime = 0
                    a = self.meals[t-1]
                    b = (950/a)**(1/(62))

                elif (t == Dtime):
                    temptime = 0
                    a = self.meals[t-1]
                    b = (Dcarb/a)**(1/(tDigest))

                elif (t == Dtime + tDigest):
                    temptime = 0
                    a = self.meals[t-1]
                    b = (950/a)**(1/(80))
                
                self.meals[t] = mealdynamics(a, b, temptime)
                #Increment temp time for the current function being run
                temptime += 1

        
        self.t = (self.t + 1) % len(self.meals)
        self.state[7] = self.state[0]       # log the previous measurements in the current state
        self.state[8] = self.state[6]
    

        time_step = np.array([0,5]) #assume measurements are taken every 5 mins
        y0 = np.array(self.state[0:6])
        meal = self.state[6]

        x = solve_ivp(self.dynamics, time_step, y0, args = (action, meal))
        #print('sim done, action was:')
        #print(action)
        for i in range(6):
            self.state[i] = x.y[i][-1]
        
        self.state[6] = self.meals[self.t]
        if self.t < self.b:
            self.last = -1
        elif self.t < self.l:
            self.last = self.t - self.b
        elif self.t < self.d:
            self.last = self.t - self.l
        else: self.last = self.t - self.d

        #print(self.state)
        return self.state

    def reward(self):
        # custom defined reward function
        if self.state[0]<=self.lower:
            reward = (self.state[0] - self.lower)*10
            return reward
        if self.lower < self.state[0] < self.target:
            reward = (self.state[0] - self.lower)
            return reward
        if self.target <= self.state[0] < self.upper:
            reward  = (self.upper - self.state[0])
            return reward
        if self.upper <= self.state[0]:
            reward = (self.upper - self.state[0])
            self.total_reward += reward
            return reward

# Updated Q Learning Function

# These dynamics and functions will be completed in the future, but I made blank ones because the QL
# will rely on them being globally defined

def qValUpdate(qtable, patient, action, alpha, gamma, lam):
    # find what the next state is going to be given the current state and action
    state1_curr = patient.state[0] #current
    state1_prev = patient.state[7] #previous
    last1 = patient.last

    # find next state
    state2 = patient.sim_action(action)
    state2_curr = state2[0]
    state2_prev = state2[7]
    last2 = patient.last

    u1 = patient.state[9]
    u2 = patient.state[10]


    # initialize quantized variables (need discrete bins for q table)
    s1_curr = 0

    #s1_prev = 0

    s2_curr = 0
    #s2_prev = 0

    # find which bins to put them in
    for s in patient.state_space:
        if abs(state1_curr - s) < abs(state1_curr - s1_curr):
            s1_curr = s

        '''if abs(state1_prev - s) < abs(state1_prev - s1_prev):
            s1_prev = s'''

        if abs(state2_curr - s) < abs(state2_curr - s2_curr):
            s2_curr = s
    
        
    action_1 = patient.state[9]
    action_2 = patient.state[10]
    q_state1 = patient.hash[(last1)]
    

    q_state2 = patient.hash[(last2)]
    
    # find the action in the next state which gives highest q
    #possible_Q = {}
    #for a in patient.action_space:
      #  possible_Q.update({a : qtable[q_state2, a]})
    #maxA = max(possible_Q, key = possible_Q.get)
    #maxQ = possible_Q[maxA]

    maxA = 0
    maxQ = 0
    for j in range(0, len(Q[0])):
        if (Q[q_state2][j] > maxQ):
            maxQ = Q[q_state2][j]
            maxA = patient.action_space[j]

    # update using the Q learning equation
    qCurrent = qtable[q_state1, np.where(patient.action_space == action)]
    qNew = (1-alpha)*qCurrent + alpha*(patient.reward() + gamma*maxQ - qCurrent)
    qtable[q_state1, np.where(patient.action_space == action)] = qNew

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
def sim_test(qtable, patient, action, alpha, gamma, lam):
    # find what the next state is going to be given the current state and action
    state1_curr = patient.state[0] #current
    state1_prev = patient.state[7] #previous
    last1 = patient.last

    # find next state
    state2 = patient.sim_action(action)
    state2_curr = state2[0]
    state2_prev = state2[7]
    last2 = patient.last

    # initialize quantized variables (need discrete bins for q table)
    s1_curr = 0
    #s1_prev = 0

    s2_curr = 0
    #s2_prev = 0

    # find which bins to put them in
    for s in patient.state_space:
        if abs(state1_curr - s) < abs(state1_curr - s1_curr):
            s1_curr = s
        '''if abs(state1_prev - s) < abs(state1_prev - s1_prev):
            s1_prev = s'''

        if abs(state2_curr - s) < abs(state2_curr - s2_curr):
            s2_curr = s
        '''if abs(state2_prev - s) < abs(state2_curr - s2_prev):
            s2_prev = s'''

    u1 = patient.state[9]
    u2 = patient.state[10]

    q_state1 = patient.hash[(last1)]
    
    q_state2 = patient.hash[(last2)]
    # find the action in the next state which gives highest q
    #possible_Q = {}
    #for a in patient.action_space:
      #  possible_Q.update({a : qtable[q_state2, a]})
    #maxA = max(possible_Q, key = possible_Q.get)
    #maxQ = possible_Q[maxA]

    maxA = 0
    maxQ = 0
    for j in range(0, len(Q[q_state2])):
        if (Q[q_state2][j] > maxQ):
            maxQ = Q[q_state2][j]   # 
            maxA = patient.action_space[j]               # find action giving highest Q value
    # update using the Q learning equation
    qCurrent = qtable[q_state1, np.where(patient.action_space == action)]
    qNew = (1-alpha)*qCurrent + alpha*(patient.reward() + gamma*maxQ - qCurrent)
    qtable[q_state1, np.where(patient.action_space == action)] = qNew

    qDif = qNew - qCurrent

    #given our next state, choose the action to take based on probability distribution
   # if (patient.state[0] < patient.lower or patient.state[0] > patient.upper):
        #action2 = maxA
    #else:
   #scores = qtable[q_state2]
    #sumQ = sum(np.exp(-lam*scores))
    #probs = np.exp(-lam*scores)/sumQ
    #action2 = random.choices(scores,probs, k=1)
    #action2 = random.choice(patient.action_space)
    #action2 = action
    return qtable, qDif, state2, maxA


# Simulation

patient1 = patient(np.zeros(11))
patient1.state[0] = 80
patient1.state[1] = 30
patient1.state[2] = 30
patient1.state[3] = 17
patient1.state[4] = 17
patient1.state[5] = 250
patient1.state[6] = 1000

patient1.state[9] = 0
patient1.state[10] = 0

'''t = 0

Q = np.zeros(((len(patient1.meals)), len(patient1.action_space)))
action = 0
while t <= 20000:
    t += 1
    Q, qDif, patient1.state, action = qValUpdate(Q, patient1, action, 0.1, 0.9999999, 0.1)
    if patient1.state[0] > 120:
        patient1.state[0] = 80
        patient1.state[1] = 30
        patient1.state[2] = 30
        patient1.state[3] = 17
        patient1.state[4] = 17
        patient1.state[5] = 250
        patient1.state[6] = 1000
'''
# Load the .npz file
data = np.load('just_bgl_random.npz')

Q = data['arr_1']

patient1.state[0] = 80
patient1.state[1] = 30
patient1.state[2] = 30
patient1.state[3] = 17
patient1.state[4] = 17
patient1.state[5] = 250
patient1.state[6] = 1000

patient1.time = []
patient1.glucose = []
patient1.actions = []
patient1.time.append(0)
action = 3
patient1.glucose.append(patient1.state[0])
patient1.actions.append(action)
t = 0
while t <= 250:
    t += 1
    patient1.time.append(t)
    Q, qDif, patient1.state, action = sim_test(Q, patient1, action, 0.1, 0.99, 0.1)

fig,ax = plt.subplots()
ax.plot(range(len(patient1.glucose)), patient1.glucose, color = "blue")
ax.set_xlabel('time (increments of 5 mins)')
ax.set_ylabel('blood glucose level (mg/dL)')

ax2 = ax.twinx()
ax2.plot(range(len(patient1.actions)), patient1.actions, color = "red")
ax2.set_xlabel('time (increments of 5 mins)')
ax2.set_ylabel('insulin dosage rate U/min)')
plt.show()