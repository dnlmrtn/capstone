def reward(self):
     if (self.lower - 45) <= self.state[0] < (self.lower - 25):
         reward = -1000
         self.total_reward += reward
         return reward
     if (self.lower - 25) <= self.state[0] < self.lower:
         reward = ((-1000)/((-25)^2))*((self.state[0] - self.lower)**2)
         self.total_reward += reward
         return reward
     if self.lower <= self.state[0] < self.target:
         reward = (self.state[0] - self.lower)**3
         self.total_reward += reward
         return reward
     if self.target <= self.state[0] < self.upper:
         reward = ((500 - (self.target - self.lower)**3)/((self.upper - self.target)**2))*((self.state[0] - self.target)**2) + ((self.target - self.lower)**3)
         self.total_reward += reward
         return reward
     if self.upper <= self.state[0] < (self.upper + 75):
         reward = ((-1000)/(75**3))*((self.state[0] - self.upper)**3) + 500
         self.total_reward += reward
         return reward
     if (self.upper + 75) <= self.state[0] <= (self.upper + 95):
         reward = -500
         self.total_reward += reward
         return reward








