# ----- BUILD ENVIRONMENT ----- #
import gym
from gym import Env
from gym.spaces import Discrete, Box
import random
import numpy as np
import scipy.integrate as integrate
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

HOURS = 16
SAMPLING_INTERVAL = 10  # min

class insulin_env(Env):
    def __init__(self):
        # actions we can take, discrete dosing size 0-10
        self.action_space  = Discrete(11)
        # BGL level
        self.observation_space = Box(low=np.float32(40), high=np.float32(160))
        

        # set insulin bounds
        self.target = 80    # target blood glucose level
        self.lower = 65     # below this level is dangerous
        self.upper = 105    # above this is dangerous
        
        # set start BGL
        self.state = []
        self.state.append(4.5)                          # plasma glucose conc. (mmol/l)
        self.state.append(15)                           # plasma insulin conc. (mu/L) in remote compartment
        self.state.append(15)                           # plasma insulin conc. (mu/L)        
        self.state.append(self.state[0])                # previous reading, will be same as current reading for initialization                  
        
        # set epoch length
        self.day_length = int(HOURS*60/SAMPLING_INTERVAL)        # awake for 16 hours, 10 mins * 96 times = 16 hours
        
        # create a list of randomly generated meals
        self.meals = [1200 + random.randint(-300,300) for i in range(int(HOURS*60/SAMPLING_INTERVAL))]
        pass
    
    def dynamics(self, t, y, u, d):
        # patient paramaters, depend on specific person
        g_basal = 4.5       # mmol/L (Note: different units from other model)
        x_basal = 15        # mu/L
        i_basal = 15        # mu/L

        # parameters for a type 1 diabetic
        v = 12                  # L
        n = 5/54                #1/min
        p1 = 0.028735           # 1/min
        p2 = 0.028344           # 1/min
        p3 = 0.00005035           # mU/L

        g = y[0]       # blood glucose measurement (mg/dL)
        x = y[1]       # remote insulin (micro-u/ml)
        i = y[2]       # insulin (micro-u/ml)

        # Compute change in state:
        dydt = np.empty(3)
        dydt[0] = -p1 * (g - g_basal) - (x - x_basal) * g + d
        dydt[1] = -p2 * (x - x_basal) + p3 * (i - i_basal)              # remote insulin compartment dynamics
        dydt[2] = -n * i + u / v
        return dydt
    
    def step(self, action):  
        # reduce day length by 1
        self.day_length -= 1
        
        # check if day is over
        if self.day_length <= 0:
            done = True
        else:
            done = False
        
        # store previous measurements
        self.state[3] = self.state[0]       # log previous measurement

        time_step = np.array([0,5])         # assume measurements are taken every 5 mins
        y0 = np.array(self.state[0:3])
        
        meal = 1
        
        # solve the ivp to get new state
        x = solve_ivp(self.dynamics, time_step, y0, args = (action, meal))
        # assign new state values

        self.state[0:3] = x.y[:,-1]
        # calculate reward
        if self.state[0]<=self.lower:                   # if below safe range
            reward =  -5
        
        if self.lower < self.state[0] < self.target:    # if safe, but below desired level
            reward = (self.state[0] - self.lower)

        if self.target < self.state[0] < self.upper:    # if safe, but above desired level
            reward  = (self.upper - self.state[0])
        
        if self.upper <= self.state[0]:                 # if above safe range
            reward = -5
        
        # set placeholder for info
        info = {}

        return self.state, reward, done, info
    def render(self):
        # implement visualization stuff later
        pass
    def reset(self):
        # reset state
        self.state = []
        self.state.append(4.5)  # plasma glucose conc. (mmol/l)
        self.state.append(15)                           # plasma insulin conc. (mu/L) in remote compartment
        self.state.append(15)                           # plasma insulin conc. (mu/L)        
        self.state.append(self.state[0])                # previous reading, will be same as current reading for initialization    
        
        self.day_length = HOURS*60/SAMPLING_INTERVAL
        return self.state

# ----- TEST ENVIRONMENT ----- #
env = insulin_env()
env.__init__()
episodes = 20
h1 = []
h2 = []
h3 = []

for episode in range(1, episodes+1):
    state,a,b,c = env.step(10)
    h1.append(state[0])
    h2.append(state[1])
    h3.append(state[2])



plt.plot(h1, label = "plasma gluc")
plt.plot(h2, label = "remote ins")
plt.plot(h3, label = "ins")
plt.legend()
plt.show()
'''
episodes = 10
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0 
    while not done:
        #env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))
'''