#Heres a basic model that Serdar was recommending

def dynamics(xt, ut):
    xnext = []
    xnext1 = (0.9)*xt[0]-0.18*ut
    xnext2 = (0.9)*xt[1]+0.07*ut
    xnext.append(xnext1)
    xnext.append(xnext2)
    return xnext

