# ----- BUILD ENVIRONMENT ----- #
import gym
from gym import Env
from gym.spaces import Discrete, Box
import random
import numpy as np
import scipy.integrate as integrate
from scipy.integrate import solve_ivp

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
        self.state.append(80 + random.randint(-15,15))  # blood glucose
        self.state.append(30)                           # remote insulin level
        self.state.append(30)                           # insulin
        self.state.append(17)                           
        self.state.append(17)                           
        self.state.append(250)                          
        self.state.append(self.state[0])                # previous reading, will be same as current reading for initialization                  
        
        # set epoch length
        self.day_length = int(HOURS*60/SAMPLING_INTERVAL)        # awake for 16 hours, 10 mins * 96 times = 16 hours
        
        # create a list of randomly generated meals
        self.meals = [1200 + random.randint(-300,300) for i in range(int(HOURS*60/SAMPLING_INTERVAL))]
        pass
    
    def dynamics(self, t, y, ui, d):
        g = self.state[0]                # blood glucose (mg/dL)
        x = self.state[1]                # remote insulin (micro-u/ml)
        i = self.state[2]                # insulin (micro-u/ml)
        q1 = self.state[3]
        q2 = self.state[4]
        g_gut = self.state[5]            # gut blood glucose (mg/dl)

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

        # Compute change in state:
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
    
    def step(self, action):  
        # reduce day length by 1
        self.day_length -= 1
        
        # check if day is over
        if self.day_length <= 0:
            done = True
        else:
            done = False
        
        # store previous measurements
        self.state[6] = self.state[0]       # previous glucose level
        
        time_step = np.array([0,10]) # assume measurements are taken every 10 mins
        y0 = np.array(self.state[0:6])
        '''
        print(int(96 - self.day_length - 1))
        print(self.meals)
        '''
        
        meal = self.meals[int(96 - self.day_length)]

        # solve the ivp to get new state
        x = solve_ivp(self.dynamics, time_step, y0, args = (action, meal))
        # assign new state values
        self.state[0:6] = x.y[:][-1]
        
        # calculate reward
        if self.state[0]<=self.lower:
            reward =  0
        if self.lower < self.state[0] < self.target:
            reward = (self.state[0] - self.lower)
        if self.target < self.state[0] < self.upper:
            reward  = (self.upper - self.state[0])
        if self.upper <= self.state[0]:
            reward = 0
        
        # set placeholder for info
        info = {}


        return self.state, reward, done, info
    def render(self):
        # implement visualization stuff later
        pass
    def reset(self):
        # reset state
        self.state[0] = 80 + random.randint(-15,15)
        self.state[1] = 30
        self.state[2] = 30
        self.state[3] = 17
        self.state[4] = 17
        self.state[5] = 250
        self.state[6] = self.state[0]
        
        self.day_length = HOURS*60/SAMPLING_INTERVAL
        return self.state
    
# ----- TEST ENVIRONMENT ----- #
env = insulin_env()
env.__init__()
print(env.state[:])
action = env.action_space.sample()
print(action)
env.step(action)
print(env.state[:])
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