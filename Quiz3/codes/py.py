import matplotlib.pyplot as plt
import numpy as np
import math 


def square(t, alpha, T, A=10):
    if 0 < t%T < alpha*T:
        return A 
    else: 
        return 0

alpha = 0.5 # duty ratio for input square wave
R = 1 # resistance
L = 1 # inductance
T = 1 # time period of input square wave

#t = np.linspace(0, n*T, int(n*T*h))

N = 1000 # Number of points of output wave to plot
h = 0.01 # Step Size


''' Forward Euler '''
fe = np.zeros(N)
t1 = np.zeros(N)
y = x = 0
for i in range(0, N):
    t1[i] = x
    fe[i] = y
    y += h*(square(x, alpha, T) - (y*R))/L
    x+=h

''' Backward Euler '''
be = np.zeros(N)
t2 = np.zeros(N)
y = x = 0
for i in range(0, N):
    be[i] = y
    t2[i] = x
    y = (y + h*square(x+h, alpha, T)/L) /(1+h*R/L)
    x+=h

''' RK 4 '''
rk4 = np.zeros(N)
t3 = np.zeros(N)
y = x = 0
for i in range(0, N):
    rk4[i] = y
    t3[i] = x
    k1 = h * (square(x, alpha, T) - y*R)/L
    k2 = h * (square(x + h/2, alpha, T) - (y+k1/2)*R)/L
    k3 = h * (square(x + h/2, alpha, T) - (y+k2/2)*R)/L
    k4 = h * (square(x + h, alpha, T) - (y+k3)*R)/L

    y += (k1 + 2*k2 + 2*k3 + k4)/6
   
    #y = (y + h*square(x+h, alpha, T)/L) /(1+h*R/L)
    x+=h


plt.plot(t1, fe, label = 'Forward Euler')
plt.plot(t2, be, label = 'Backward Euler')
plt.plot(t3, rk4, label = 'Colorful Dog 4')
plt.legend()
plt.grid()
plt.show()

