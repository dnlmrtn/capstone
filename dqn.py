import random
import numpy as np
import scipy.integrate as integrate
from scipy.integrate import solve_ivp

class patient:   # initializing patient environment
    def __init__(self, state):
        self.t = 0
        self.state = state # initialize the state of a human body

        # define the state space and action space
        self.state_space = np.linspace(0, 10, 11)
        
        # To decrease state space size, BGL is quantized into 11 bins:
        # [40-50 50-60, 60-65 65-70 75-80 80-85 85-90 90-95 95-100 100-110 110-120] mg/dL
        self.action_space = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10) # possible dose sizes

        self.target = 80    # target blood glucose level
        self.lower = 65     # below this level is dangerous, NO insulin should be administered
        self.upper = 105    # above this is dangerous, perceived optimal dose must be administered

    def dynamics(self, t, y, ui, d):
        g = y[0]                # blood glucose (mg/dL)
        x = y[1]                # remote insulin (micro-u/ml)
        i = y[2]                # insulin (micro-u/ml)
        q1 = y[3]               # 
        q2 = y[4]               # 
        g_gut = y[5]            # gut blood glucose (mg/dl)

        # Body parameters:
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

        # Linearized dynamical system:
        dydt = np.empty(6)
        dydt[0] = -p1*(g-gb) - si*x*g + f*kabs/vg * g_gut + f/vg * d    # blood glucose (mg/dL)
        dydt[1] =  p2*(i-x)                                             # remote insulin compartment dynamics
        dydt[2] = -ke*i + ui                                            # insulin
        dydt[3] = ui - kemp * q1
        dydt[4] = -kemp*(q2-q1)
        dydt[5] = kemp*q2 - kabs*g_gut

        # convert from minutes to hours
        dydt = dydt*60
        return dydt

    def sim_action(self, action):
        if self.state is None:
            raise Exception("Please reset() environment")

        self.state[7] = self.state[0]
        self.state[8] = self.state[6]

        time_step = np.array([0,10])    # assume measurements are taken every 10 mins
        meal = 1100 + np.random.random()*200   # generate a random blood sugar intake

        # solve ivp and return new state
        x = solve_ivp(self.dynamics, time_step, np.array(self.state[0:6]), args = (action, meal))
        print(x.y[0][-1])
        return self.state

    def reward(self):
        if self.state[0]<=self.lower:
            return 0
        if self.lower < self.state[0] < self.target:
            reward = (self.state[0] - self.lower)
            return reward
        if self.target < self.state[0] < self.upper:
            reward  = (self.upper - self.state[0])
            return reward
        if self.upper <= self.state[0]:
            return 0

patient1 = patient(np.zeros(9))
patient1.state[0] = 90
patient1.state[1] = 30
patient1.state[2] = 30
patient1.state[3] = 17
patient1.state[4] = 17
patient1.state[5] = 250
patient1.state[6] = 0

print(patient1.state[0])
patient1.sim_action()

# Build the DQN
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import Adam   

from collections import deque

import time
import numpy as np
import random

import tensorflow as tf

REPLAY_MEMORY_SIZE = 50_000
MODEL_NAME = '256x2'
MINIBATCH_SIZE = 10000
DISCOUNT = 0.99

# Own Tensorboard class # Changes how often we create a log file
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

class DQNagent:
    def __init__(self):
        # main model # trained every step
        self.model = self.create_model()
        
        # target model # what we predict against every step 
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.tensorboard = ModifiedTensorBoard(log_dir = f"logs/{MODEL_NAME}-{int(time.time())}" )
        
        self.target_upodate_counter = 0
        
    def create_model(self):
        model = Sequential()
        
        model.add(Conv2D(256, (3,3), input_shape = patient.OBSERVATION_INPUT_SHAPE))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2,2))
        model.add(Dropout(0.2))
        
        model.add(Conv2D(256, (3,3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2,2))
        model.add(Dropout(0.2))
        
        model.add(Flatten())
        model.add(Dense(64))
        
        model.add(Dense(patient.ACTIVATION_SPACE_SIZE))
        model.compile(loss='mse', optimizer=Adam(lr=0.001),metrics=['accuracy'])
        
        return model
    
    def update_replay_memory(self, transition):
        self.replay_memory.appen(transition)
        
    def get_qs(self, state, step):
        return self.model_predict(np.array(state).reshape(-1, *state.shape)/255)[0]
    
    def train(self,terminal_state, step):
        if len(self.replay_memory) < REPLAY_MEMORY_SIZE:
            return
        
        # take a small group of random samples(minibatch) from memory replay
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        
        # get the current states from minibatch, pass through NN for Q values
        current_states = np.array([transition[0] for transition in minibatch])/255
        current_qs_list = self.model.predict(current_states)
        
        # get future states from minibatch, pass through NN for Q values
        new_current_states = np.array([transition[3] for transition in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states)
        
        X = []
        y = []
        
        gym.spaces
        
        # enumerate the minibatch
        for index, (current_state, action, reward, new_current_states, done) in enumerate(minibatch):
            # if not in a terminal state, get new Q from future states, otherwise set it to 0
            # similar to Q learning, but we use bellman equation here
            if not done:
                max_future_q = max(future_qs_list)
                new_q = reward + discount*max_future_q
            else:
                new_q = reward
                
            #update q values for the given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q
            X.append(current_state)
            y.append(current_qs)
            
        # Fit on all samples as one batch
        
        
from gym import envsfrom gym.spaces import Discrete, Box
import numpy as np
import random
env = gym.make('CartPole-v0')
states = env.observation_space.shape[0]
actions = env.action_space.n

actions