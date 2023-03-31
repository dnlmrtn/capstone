# 
import sympy as sp
import numpy as np

g, x, i, q1, q2, g_gut, ui, d = sp.symbols('g x i q1 q2 g_gut ui d')

params = [g, x, i, q1, q2, g_gut]


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


f1 = -p1*(g-gb) - si*x*g + f*kabs/vg * g_gut + f/vg * d
f2 =  p2*(i-x) # remote insulin compartment dynamics
f3 = -ke*i + ui # insulin dynamics
f4 = ui - kemp * q1
f5 = -kemp*(q2-q1)
f6 = kemp*q2 - kabs*g_gut

functions = [f1, f2, f3, f4, f5, f6]

J = np.zeros((6,6))
derivatives = []
for func in functions:
    for var in params:
        derivatives.append(sp.diff(func,var))

lin = []

for i in range(len(derivatives)):
    lin.append(derivatives[i].sp.subs)

