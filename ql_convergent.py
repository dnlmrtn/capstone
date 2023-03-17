import random
import numpy as np
import scipy.integrate as integrate
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
# #import seaborn as sn

def mealdynamics(a, b, t):
            
            meal = a * (b ** t)

            return meal

class patient:
    def __init__(self, state, prev_g_readings=int, prev_u_readings=int, indicator=bool, time=bool):

        # These variables tell us how many
        self.num_prev_g = prev_g_readings
        self.num_prev_u = prev_u_readings
        self.t = 0
        self.last = -1

        if time:
            self.has_time = True
        if indicator:
            self.has_indicator = True

        # add dynamics variables to state
        self.state = state

        # add state readings for previous glucose and control
        self.state.append([0] * (prev_g_readings + prev_u_readings))

        # add meal indicator if needed
        if indicator:
            self.state.append([0])
        # add time indicator if needed
        if time:
            self.state.append([0])
        
        self.actions = []
        self.glucose = []
        self.mealamount = []
        self.time = []
        self.state_space = np.linspace(0, 250, 30)
        self.action_space = range(11) #possible doses

        self.meal_space = np.linspace(800, 2000, 10) 

        self.target = 80    #target blood glucose level
        self.lower = 65     #below this level is dangerous, NO insulin should be administered
        self.upper = 105    #above this is dangerous, perceived optimal dose must be administered
        self.b = 10
        self.l = 80
        self.d = 150
        
        self.meals = [0]*203

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

        return dydt*60
    

    def sim_action(self, action):

        # get the meal
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
                    a = 1000
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
                patient1.mealamount.append(self.meals[t])

        self.t = (self.t + 1) % len(self.meals)

        # measurements are taken every 5 mins
        time_step = np.array([0,5]) 

        # assign initial cond for ivp
        y0 = np.array(self.state[0:6])

        # assign the meal
        meal = self.meals[t]

        # solve the ivp
        x = solve_ivp(self.dynamics, time_step, y0, args = (action, meal))

        # log the previous glucose readings       
        if self.num_prev_g == 1:
            self.state[6] = self.state[1]
        
        if self.num_prev_g == 2:
            self.state[5+2] = self.state[5+1]
            self.state[5+1] = self.state[1]
        
        # log the previous insulin doses
        if self.num_prev_u==1:
            self.state[5 + self.num_prev_g + 1] = action

        if self.num_prev_u==2:
            self.state[5 + self.num_prev_g + 2] = self.state[5 + self.num_prev_g + 1]
            self.state[5 + self.num_prev_g + 1] = action

        # update dynamic state values
        for i in range(6):
            self.state[i] = x.y[i][-1]

        # mealtime assignments
        if self.t < self.b:
            self.last = -1
        elif self.t < self.l:
            self.last = self.t - self.b
        elif self.t < self.d:
            self.last = self.t - self.l
        else: self.last = self.t - self.d

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
            return reward

# Updated Q Learning Function

# These dynamics and functions will be completed in the future, but I made blank ones because the QL
# will rely on them being globally defined

def qValUpdate(qtable, patient, action, gamma):
    '''
    Steps:
    Quantize patient state
    find best action
    '''
    if patient.time:
        time_dimensions = 1
    else:
        time_dimensions = 0
    
    if patient.indicator:
        indicator_dimensions = 1
    else:
        indicator_dimensions = 0

    observable_dimensions = 1 + patient.num_prev_g + patient.num_prev_u + time_dimensions + indicator_dimensions

    quantized_state = quantize(patient.state)

    # create a key based on all of the readings
    key = f"{patient.state[0]}"
    for i in range(patient.num_prev_g):
        index = index + f"{quantize(patient.state[5 + i + 1])}"
    
    for i in range(patient.num_prev_u):
        index = index + f"{patient.state[5 + patient.num_prev_g + i + 1]}"

    if patient.has_time:
        index = index + f"{patient.state[5 + patient.num_prev_g + patient.num_prev_u + 1]}"

    if patient.has_indicator:
        if patient.has_time:
            index = index + f"{patient.state[5 + patient.num_prev_g + patient.num_prev_u + 2]}"
        else:
            index = index + f"{patient.state[5 + patient.num_prev_g + patient.num_prev_u + 1]}"

    # turn it into an integer
    index = int(key)

    entry_state = patient.state

    # find what the next state is going to be given the current state and action
    current_glucose = patient.state[0] #current

    # find next state
    exit_state = patient.sim_action(action)

    Q = np.zeros()
    action_1 = patient.state[9]
    action_2 = patient.state[10]
    q_state1 = patient.hash[(s1_quantized, last1)]
    
    q_state2 = patient.hash[(s2_quantized, last2)]
    
    # find the action in the next state which gives highest q

    maxA = 0
    maxQ = 0
    for j in range(0, len(Q[0])):
        if (qtable[q_state2, j][0] > maxQ):
            maxQ = qtable[q_state2, j][0]
            maxA = j

    # Get current Q value
    qCurrent = qtable[q_state1, action][0]

    # Update no. of state visits
    alpha = 1/qtable[q_state1, action][1]       # defines learning rate
    qtable[q_state1, action][1] += 1

    # Update using the Q learning equation
    qNew = (1-alpha)*qCurrent + alpha*(patient.reward() + gamma*maxQ - qCurrent)
    qtable[q_state1, action][0] = qNew

    # Get difference of Q values
    qDif = qNew - qCurrent

    action2 = random.choice(patient.action_space)

    return qtable, qDif, state2, action2


# Meals
def sim_test(qtable, patient, action):
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
    s1_quantized = 0
    s2_quantized = 0

    # find which bins to put them in
    for s in patient.state_space:
        if abs(state1_curr - s) < abs(state1_curr - s1_quantized):
            s1_quantized = s

        if abs(state2_curr - s) < abs(state2_curr - s2_quantized):
            s2_quantized = s

    action_1 = patient.state[9]
    action_2 = patient.state[10]

    q_state1 = patient.hash[(s1_quantized, last1)]
    q_state2 = patient.hash[(s2_quantized, last2)]

    # find the action in the next state which gives highest q
    maxA = 0
    maxQ = 0
    for j in range(0, len(Q[q_state2])):
        if (qtable[q_state2][j][0] > maxQ):
            maxQ = qtable[q_state2][j][0]   # 
            maxA = j                # find action giving highest Q value
    


    #Cross-Validation - output reward from function
    reward = patient.reward()

    return state2, maxA, reward

# Simulation
g_readings = 1
u_readings = 1

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

t = 0

Q = np.zeros((len(patient1.state_space)*len(patient1.meals), len(patient1.action_space), 2))
action = 0
while t <= 1000:
    t += 1
    Q, qDif, patient1.state, action = qValUpdate(Q, patient1, 0.9999999, 0.1)

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
totalReward = 0
while t <= 500:
    t += 1
    patient1.time.append(t)
    Q, qDif, patient1.state, action, reward = sim_test(Q, patient1, action, 0.1, 0.99, 0.1)
    #Cross-Validation - Track total reward for example
    totalReward += reward

fig,ax = plt.subplots()
ax.plot(range(len(patient1.glucose)), patient1.glucose, color = "blue")
ax.set_xlabel('time (increments of 5 mins)')
ax.set_ylabel('blood glucose level (mg/dL)')

ax2 = ax.twinx()
ax2.plot(range(len(patient1.actions)), patient1.actions, color = "red")
ax2.set_xlabel('time (increments of 5 mins)')
ax2.set_ylabel('insulin dosage rate U/min)')
plt.show()

plt.plot(range(len(patient1.mealamount)), patient1.mealamount)
plt.show()

print(totalReward)