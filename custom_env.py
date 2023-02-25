# ----- BUILD ENVIRONMENT ----- #
import gym
from gym import Env
from gym.spaces import Discrete, Box
import random
import numpy as np
import scipy.integrate as integrate
from scipy.integrate import solve_ivp
import pygame
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import os
import random
from collections import deque, namedtuple
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

HOURS = 16
SAMPLING_INTERVAL = 5  # min

class insulin_env(Env):
    def __init__(self):
        # actions we can take, discrete dosing size 0-10
        self.action_space  = Discrete(10)
        # BGL level
        self.observation_space = Box(low=np.float32(40), high=np.float32(160))
        
        # set insulin bounds
        self.target = 80    # target blood glucose level
        self.lower = 65     # below this level is dangerous
        self.upper = 105    # above this is dangerous
        
        # set start BGL
        self.state = []
        self.state.append(80)                           # blood glucose
        self.state.append(30)                           # remote insulin level
        self.state.append(30)                           # insulin
        self.state.append(17)                           
        self.state.append(17)                           
        self.state.append(50)                                          
        self.state.append(0)                            # time
        # set epoch length
        self.day_length = int(HOURS*60/SAMPLING_INTERVAL)        # awake for 16 hours, 10 mins * 96 times = 16 hours
        
        
        # create a list of randomly generated meals
        #self.meals = [500*np.sin(i/300) + random.randint(-300,300) for i in range(int(HOURS*60/SAMPLING_INTERVAL))]
        self.meals = [900 + 200*(1 + np.sin(i/12)) + random.randint(-200,200) for i in range(int(HOURS*60/SAMPLING_INTERVAL))]
        
        pass
    
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
        
        g = y[0]        # blood glucose measurement (mg/dL)
        x = y[1]        # remote insulin (micro-u/ml)
        i = y[2]        # insulin (micro-u/ml)
        q1 = y[3]
        q2 = y[4]
        g_gut = y[5]    # gut blood glucose (mg/dl)
        
        # Compute change in state:
        dydt = np.empty(6)
        dydt[0] = -p1*(g-gb) - si*x*g + f*kabs/vg * g_gut + f/vg * d
        dydt[1] =  p2*(i-x) # remote insulin compartment dynamics
        dydt[2] = -ke*i + ui # insulin dynamics
        dydt[3] = ui - kemp * q1
        dydt[4] = -kemp*(q2-q1)
        dydt[5] = kemp*q2 - kabs*g_gut
        # dydt[6] =         placeholder for previous glucose measurements

        return 60*dydt
    
    def step(self, action):  
        # intrease by 1
        self.state[6] += 1
        
        # check if day is over
        if self.day_length <= self.state[6]:
            done = True
        else:
            done = False
    
        time_step = np.array([0,5]) # assume measurements are taken every 5 min
        y0 = np.array(self.state[0:6])
        
        meal = self.meals[self.state[6] % len(self.meals)]

        # solve the ivp to get new state
        x = solve_ivp(self.dynamics, time_step, y0, args = (action, meal))

        # assign new state values
        self.state[0:6] = x.y[:,-1]
        
        # calculate reward
        if self.state[0] <= self.lower:
            reward =  0
            
        if self.lower < self.state[0] < self.target:
            reward = (self.state[0] - self.lower)
            
        if self.target < self.state[0] < self.upper:
            reward  = (self.upper - self.state[0])
            
        if self.upper <= self.state[0]:
            reward = 0
        
        # set placeholder for info
        info = {}
        # print("bgl: ", self.state[0])
        # print("reward: ", reward)
        
        return self.state, reward, done

    def render(self, xs, ys):
        pass
    def reset(self):
        # reset state
        self.state[0] = 80
        self.state[1] = 30
        self.state[2] = 30
        self.state[3] = 17
        self.state[4] = 17
        self.state[5] = 250
        #self.state[6] = self.state[0]
        
        self.day_length = HOURS*60/SAMPLING_INTERVAL
        return self.state
    
# ----- TEST ENVIRONMENT ----- #
env = insulin_env()
env.__init__()
env = insulin_env()
env.__init__()

style.use('fivethirtyeight')

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

iters = env.day_length

idx = list(range(iters))
bgl_memory = []
for i in range(iters):
    if env.state[0] < env.lower:
        env.step(0)
    if  env.lower <= env.state[0] and env.state[0] <= env.target:
        env.step(1)
    if  env.target <= env.state[0] and env.state[0] <= env.upper:
        env.step(5)
    if  env.upper < env.state[0]:
        env.step(10)
    bgl_memory.append(env.state[0])

plt.subplot(1,2,1)
plt.plot(idx, bgl_memory, label = "meals")
plt.subplot(1,2,2)
plt.plot(idx, env.meals)

plt.show()
'''


#import cv2
import gym
import numpy as np
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim
'''
states = 1
actions = 11

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NAME = "vanilla-update-every-4-steps"
SAVE_FREQ = 25
ACTIONS = [list(range(10))
]
# --- building the DQN ---
class DQN(nn.Module):
    def __init__(
        self,
        ninputs,
        noutputs,
        seed=None,
    ):
        super(DQN, self).__init__()

        if seed:
            self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(ninputs, 432)
        self.fc1 = nn.Linear(432, 216)
        self.fc3 = nn.Linear(216, noutputs)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fx2(x)
        x = self.fc3(x)
        return x

    def predict(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        self.eval()
        with torch.no_grad():
            pred = self.forward(state)

        return pred

"""Fixed-size buffer to store experience tuples."""
class ReplayBuffer:

    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = (
            torch.from_numpy(np.array([e.state for e in experiences if e is not None]))
            .float()
            .to(device)
        )
        actions = (
            torch.from_numpy(np.array([e.action for e in experiences if e is not None]))
            .long()
            .to(device)
        )
        rewards = (
            torch.from_numpy(np.array([e.reward for e in experiences if e is not None]))
            .float()
            .to(device)
        )
        next_states = (
            torch.from_numpy(
                np.array([e.next_state for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        dones = (
            torch.from_numpy(np.array([e.done for e in experiences if e is not None]))
            .float()
            .to(device)
        )
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.memory)
    
class insulinAgent:
    def __init__(
        self,
        actions=ACTIONS,
        gamma=0.95,  # discount rate
        epsilon=1.0,  # random action rate
        epsilon_min=0.1,
        epsilon_decay=0.9999,
        learning_rate=0.001,
        tau=1e-3,  # soft update discount
        update_main_network_freq=1,
        hard_update=False,
        dqn_loss="mse",
        act_interval=2,
        buffer_size=5000,
        batch_size=64,
        save_freq=25,
        seed=None,
    ):
        self.obs_shape = 2
        self.actions = actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.tau = tau
        self.update_main_network_freq = update_main_network_freq
        self.hard_update = hard_update
        self.dqn_loss = dqn_loss

        self.seed = seed if seed is not None else np.random.randint(1000)
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.dqn_behavior = DQN(3, len(self.actions), self.seed).to(device)
        self.dqn_target = DQN(3, len(self.actions), self.seed).to(device)
        self.optimizer = optim.Adam(self.dqn_behavior.parameters(), lr=learning_rate)

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.memory = ReplayBuffer(
            action_size=len(self.actions),
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            seed=seed,
        )

        self.act_interval = act_interval
        self.save_freq = save_freq
        self.training_steps = 0

    def act(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() > self.epsilon:
            action_values = self.dqn_behavior.predict(state)
            aind = np.argmax(action_values.cpu().data.numpy())

        else:
            aind = random.randrange(len(self.actions))

        return self.actions[aind]

    def soft_update_target(self):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1-τ)*θ_target
        """
        for target_param, local_param in zip(
            self.dqn_target.parameters(), self.dqn_behavior.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

    def learn(self):
        states, actions, rewards, next_states, dones = self.memory.sample()

        # get Q tables from both networks
        q_targets_next = self.dqn_target(next_states).detach().max(1)[0]
        q_targets = (rewards + self.gamma * q_targets_next * (1 - dones)).unsqueeze(1)

        q_preds = self.dqn_behavior(states)
        q_preds = q_preds.gather(1, actions.unsqueeze(1))

        # fit behavior dqn
        self.dqn_behavior.train()
        self.optimizer.zero_grad()
        if self.dqn_loss == "mse":
            loss = F.mse_loss(q_preds, q_targets)
        elif self.dqn_loss == "huber":
            loss = F.huber_loss(q_preds, q_targets)
        loss.backward()
        self.training_steps += 1

        # Frequency at which main network weights should be updated
        if self.training_steps % self.update_main_network_freq == 0:
            self.optimizer.step()

        if not self.hard_update:
            self.soft_update_target()
        else:
            self.hard_update_target()

        # decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def load(self, fp):
        checkpoint = torch.load(fp)
        self.dqn_behavior.load_state_dict(checkpoint["model_state"])
        self.dqn_target.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

    def save(self, epoch, steps, reward, epsilon, loss):
        print(f"saving model to models/{NAME}-{epoch}.pth")

        fp = f"{NAME}-hist.csv"

        if not os.path.exists(fp):
            with open(fp, "w") as f:
                f.write(f"epoch,epsilon,steps,reward,loss\n")

        with open(fp, "a") as f:
            f.write(f"{epoch},{epsilon},{steps},{reward},{loss}\n")

        torch.save(
            {
                "model_state": self.dqn_target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            f"models/{NAME}-{epoch}.pth",
        )

    def train(self, env: gym.Env, start_ep: int, end_ep: int, max_neg=25):
        print(f"Starting training from ep {start_ep} to {end_ep}...")
        for ep in range(start_ep, end_ep + 1):
            state = env.reset()

            total_reward = 0
            n_rewards = 0
            state_queue = deque([state] * 3, maxlen=3)  # queue 3 states
            t = 1
            done = False

            while True:
                state_stack = np.array(state_queue)
                action = self.act(state_stack)

                reward = 0
                for _ in range(self.act_interval + 1):
                    next_state, r, done = env.step(action)
                    reward += r
                    if done:
                        break

                # end episode if continually getting negative reward
                n_rewards = n_rewards + 1 if t > 100 and reward < 0 else 0

                total_reward += reward

                next_state = process_state(next_state)  # type: ignore
                state_queue.append(next_state)
                next_state_stack = np.array(state_queue)

                self.memory.add(
                    state_stack,
                    self.actions.index(action),
                    reward,
                    next_state_stack,
                    done,
                )

                if done or n_rewards >= max_neg or total_reward < 0:
                    print(
                        f"episode: {ep}/{end_ep}, length: {t}, total reward: {total_reward:.2f}, epsilon: {self.epsilon:.2f}"
                    )
                    break

                if len(self.memory) > self.batch_size:
                    loss = self.learn()

                t += 1

            if ep % self.save_freq == 0:
                self.save(ep, t, total_reward, epsilon, loss)
                
def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        help="Path to partially trained model (hd5)",
    )
    parser.add_argument(
        "-s",
        "--start",
        type=int,
        default=1,
        help="starting episode (to continue training from)",
    )
    parser.add_argument("-x", "--end", type=int, default=1000, help="ending episode")
    parser.add_argument(
        "-e",
        "--epsilon",
        type=float,
        default=1.0,
        help="Starting epsilon (default: 1)",
    )

    args = parser.parse_args()

    if args.model:
        print("loading a model, make sure start and epsilon are set correctly")

    return args.model, args.start, args.end, args.epsilon

if __name__ == "__main__":
    model_path, start, end, epsilon = get_args()

env = insulin_env()
env.__init__()
agent = insulinAgent(
        actions=ACTIONS,
        gamma=0.95,  # discount rate
        epsilon=epsilon,  # random action rate
        epsilon_min=0.1,
        epsilon_decay=0.9999,
        learning_rate=0.001,
        tau=1e-3,  # soft update discount
        update_main_network_freq=1,
        hard_update=False,
        dqn_loss="mse",
        act_interval=2,
        buffer_size=5000,
        batch_size=64,
        save_freq=SAVE_FREQ,
        seed=420,
    )

if model_path:
    agent.load(model_path)

# agent.train(env, start, end)

env.close()
'''