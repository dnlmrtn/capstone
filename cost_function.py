
def reward(self):
        if self.state[0]<self.lower:
            return 0
        if self.lower <= self.state[0] <= self.target:
            reward = (self.state[0] - self.lower)^3
            return reward
        if self.target < self.state[0] <= self.upper:
            reward  = -((self.target-self.lower)^3/(self.upper-self.target)^2)*((self.state[0]-self.target)^2)+((self.target-self.lower)^3)
            return reward
        if self.upper < self.state[0]:
            return 0








