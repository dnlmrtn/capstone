#Heres a basic model that Serdar was recommending
import numpy as np
from scipy.integrate import solve_ivp

#state definition
#x[0] = Blood Glucose (mg/dL)
#x[1] = Remote Insulin (micro-u/ml)
#x[2] = Plasma Insulin (micro-u/ml)
#x[3] = S1
#x[4] = S2
#x[5] = gut Blood Glucose (mg/dl)
#x[6] = Meal Disturbance (mmol/L-min)
#x[7] = Previous Blood Glucose (mg/dL)
#x[8] = Previous meal disturbance (mmol/L-min)

def __init__(self):
    self.state = None #state
    self.u = None #actions taken (not used by algorithm, will be used for plotting after)

    self.action_space = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] #possible doses

    self.target = 80 #target blood glucose level
    self.lower = 65 #below this level is dangerous, NO insulin should be administered
    self.upper = 105 #above this is dangerous, perceived optimal dose must be administered

def sim_action(self, action):

    if self.state is None:
        raise Exception("Please reset() environment")
    
    self.u.append(action)

    time_step = [0, 10] #assume measurements are taken every 10 mins

    x_next = solve_ivp(dynamics, time_step, self.state)

    for i in range(10):
        self.state[i] = x_next[i][10]

    return self.state

def dynamics(state, t, action, meal):
    g = state[0] #blood glucose (mg/dL)
    x = state[1] #remote insulin (micro-u/ml)
    i = state[2] #plasma insulin (micro-u/ml)
    q1 = state[3] #S1
    q2 = state[4] #S2
    g_gut = state[5] #gut blood glucose (mg/dL)

    #parameters (??)
    gb = 291.0     # (mg/dL)                    Basal Blood Glucose
    p1 = 3.17e-2   # 1/min
    p2 = 1.23e-2   # 1/min
    si = 2.9e-2    # 1/min * (mL/micro-U)
    ke = 9.0e-2    # 1/min                      Insulin elimination from plasma
    kabs = 1.2e-2  # 1/min                      t max,G inverse
    kemp = 1.8e-1  # 1/min                      t max,I inverse
    f = 8.00e-1    # L
    vi = 12.0      # L                          Insulin distribution volume
    vg = 12.0      # L                          Glucose distibution volume

     # Compute ydot:
    dydt = np.empty(6)

    dydt[0] = -p1 * (g - gb) - si * x * g + f * kabs / vg * g_gut + f / vg * meal  # (1)
    dydt[1] = p2 * (i - x)  # remote insulin compartment dynamics (2)
    dydt[2] = -ke * i + action  # plasma insulin concentration  (3)
    dydt[3] = action - kemp * q1  # two-part insulin absorption model dS1/dt
    dydt[4] = -kemp * (q2 - q1)  # two-part insulin absorption model dS2/dt
    dydt[5] = kemp * q2 - kabs * g_gut

    # convert from minutes to hours
    dydt = dydt * 60
    return dydt