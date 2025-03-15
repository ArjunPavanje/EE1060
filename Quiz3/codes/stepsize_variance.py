import matplotlib.pyplot as plt
import numpy as np
import math 
from matplotlib.widgets import Button, Slider
from matplotlib.widgets import RangeSlider

N = 10 # Number of points of output wave to plot
h = 0.01 # Step Size

def square(t, alpha, T, A=10):
    if 0 < t%T < alpha*T:
        return A 
    else: 
        return 0


def backward_euler(T, R, L, alpha, h):
    fe = []
    t1 = []
    y = x = 0
    for i in t:
        fe.append(y)
        t1.append(x)
        #np.append(fe, y)
        #np.append(t1, x)
        #t1[i] = x
        #fe[i] = y
        y = ((L*y) + (h*square(x+h, alpha, T)) )/(L+h*R)
        x+=h 
    return fe

alpha = 0.5 # duty ratio for input square wave
R = 1 # resistance
L = 1 # inductance
T = 1 # time period of input square wave

# h = 0.01
t = np.linspace(0, N*T, int(N*T/h))
plt.plot(t, backward_euler(T, R, L, alpha, h), label = "h = 0.01")

# h = 0.1
h = 0.1
t = np.linspace(0, N*T, int(N*T/h))
plt.plot(t, backward_euler(T, R, L, alpha, h), label = "h = 0.1")


# h = 0.05
h = 0.05
t = np.linspace(0, N*T, int(N*T/h))
plt.plot(t, backward_euler(T, R, L, alpha, h), label = "h = 0.05")


plt.legend()
plt.show()

