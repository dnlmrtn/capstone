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
        self.state.append(30)
        self.state.append(5)                           # insulin                      
        self.state.append(self.state[0])                # previous reading, will be same as current reading for initialization                  
        
        # set epoch length
        self.day_length = int(HOURS*60/SAMPLING_INTERVAL)        # awake for 16 hours, 10 mins * 96 times = 16 hours
        
        # create a list of randomly generated meals
        self.meals = [1200 + random.randint(-300,300) for i in range(int(HOURS*60/SAMPLING_INTERVAL))]
        pass
    
    def dynamics(self, t, y, u, d):
        g = self.state[0]       # blood glucose (mg/dL)
        x = self.state[1]       # remote insulin (micro-u/ml)
        i = self.state[2]       # insulin (micro-u/ml)
        gsc = self.state[3]     
        ggut = self.state[4]    

        # Parameters:
        gb = 81.3               # Basal Blood Glucose (mg/dL)
        gbsc = 77.7             # mg/dL
        ib = 15                 # Basal insulin Concentration
        v = 12                  # L
        n = 5/54                #1/min

        # Bergman Parameters
        p1 = 0.028735           # 1/min
        p2 = 0.0283444          # 1/min
        p3 = 5.035e-5           # mU/L

        # Compute change in state:
        dydt = np.empty(3)
        dydt[0] = -p1*g - x*(g + gb) + d
        dydt[1] =  p2*x + p3*i # remote insulin compartment dynamics
        dydt[2] = n*(i + ib) + u/v
        dydt[3] = (g-gsc)/5 - R     # What is R? Still working on this part lol

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