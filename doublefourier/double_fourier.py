import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import nquad


wm = 5 # sin wave's frequency
m = 4 # f_c = m f_m, m in N
wc = m * wm # triangular wave's frequency 

Ac = 1 # sin wave's amplitude
mf = 0.1 # Ac = mf Am, mf < 1
Am = mf * Ac 

def carrier(A, w, t): # triangular wave
    return (A)-(2*A/(np.pi)*np.arccos(np.cos(w*t -(np.pi/2))))

def modulating(A, w, t): # sin wave
    return A*np.sin(w*t)


def double_fourier(out, t, T1, T2):
    N = M = 10  # Number of harmonics to consider
    
    def f(t1, t2):
        idx = int(t1 / (T1 / len(t)))
        if idx >= len(out):
            idx = len(out) - 1
        return out[idx]
  
    x_vals = np.linspace(0, T1, 100)  
    y_vals = np.linspace(0, T2, 100)
    inner_integral = [np.trapezoid([f(x, y) for x in x_vals], dx=(x_vals[1] - x_vals[0])) for y in y_vals]
    a00 = np.trapezoid(inner_integral, dx=(y_vals[1] - y_vals[0]))

    #a00 = 1/(T1*T2) * nquad(f, [(0, T1), (0, T2)], opts = [{'limit':10000}, {'limit':10000} ] )[0]

    def calculate_coefficient(m, n, coeff_type):
        if coeff_type == 'a0n':
            x_vals = np.linspace(0, T1, 100)  
            y_vals = np.linspace(0, T2, 100)
            inner_integral = [np.trapezoid([f(x, y)*np.cos(2*np.pi*n*y/T2) for x in x_vals], dx=(x_vals[1] - x_vals[0])) for y in y_vals]
            return np.trapezoid(inner_integral, dx=(y_vals[1] - y_vals[0]))
            #return (2 / T1*T2) * np.trapezoid([f(0, t2) * np.cos(2 * np.pi * n * t2 / T2) for t2 in np.linspace(0, T2, 100)], dx=T2/100)

        elif coeff_type == 'b0n':
            x_vals = np.linspace(0, T1, 100)  
            y_vals = np.linspace(0, T2, 100)
            inner_integral = [np.trapezoid([f(x, y)*np.sin(2*np.pi*n*y/T2) for x in x_vals], dx=(x_vals[1] - x_vals[0])) for y in y_vals]
            return np.trapezoid(inner_integral, dx=(y_vals[1] - y_vals[0]))

        elif coeff_type == 'am0':
            x_vals = np.linspace(0, T1, 100)  
            y_vals = np.linspace(0, T2, 100)
            inner_integral = [np.trapezoid([f(x, y)*np.cos(2*np.pi*m*x/T1) for x in x_vals], dx=(x_vals[1] - x_vals[0])) for y in y_vals]
            return np.trapezoid(inner_integral, dx=(y_vals[1] - y_vals[0]))

        elif coeff_type == 'bm0':
            x_vals = np.linspace(0, T1, 100)  
            y_vals = np.linspace(0, T2, 100)
            inner_integral = [np.trapezoid([f(x, y)*np.sin(2*np.pi*m*x/T1) for x in x_vals], dx=(x_vals[1] - x_vals[0])) for y in y_vals]
            return np.trapezoid(inner_integral, dx=(y_vals[1] - y_vals[0]))        

        elif coeff_type == 'cmn':
            x_vals = np.linspace(0, T1, 100)  
            y_vals = np.linspace(0, T2, 100)
            inner_integral = [np.trapezoid([f(x, y)*np.cos(2*np.pi*m*x/T1)*np.sin(2*np.pi*n*x/T2) for x in x_vals], dx=(x_vals[1] - x_vals[0])) for y in y_vals]
            return np.trapezoid(inner_integral, dx=(y_vals[1] - y_vals[0]))

        elif coeff_type == 'dmn':
            x_vals = np.linspace(0, T1, 100)  
            y_vals = np.linspace(0, T2, 100)
            inner_integral = [np.trapezoid([f(x, y)*np.sin(2*np.pi*m*x/T1)*np.cos(2*np.pi*n*x/T2) for x in x_vals], dx=(x_vals[1] - x_vals[0])) for y in y_vals]
            return np.trapezoid(inner_integral, dx=(y_vals[1] - y_vals[0]))

        elif coeff_type == 'amn':
            x_vals = np.linspace(0, T1, 100)  
            y_vals = np.linspace(0, T2, 100)
            inner_integral = [np.trapezoid([f(x, y)*np.cos(2*np.pi*m*x/T1)*np.cos(2*np.pi*n*x/T2) for x in x_vals], dx=(x_vals[1] - x_vals[0])) for y in y_vals]
            return np.trapezoid(inner_integral, dx=(y_vals[1] - y_vals[0]))

        elif coeff_type == 'bmn':
            x_vals = np.linspace(0, T1, 100)  
            y_vals = np.linspace(0, T2, 100)
            inner_integral = [np.trapezoid([f(x, y)*np.sin(2*np.pi*m*x/T1)*np.sin(2*np.pi*n*x/T2) for x in x_vals], dx=(x_vals[1] - x_vals[0])) for y in y_vals]
            return np.trapezoid(inner_integral, dx=(y_vals[1] - y_vals[0]))

    a0n = [calculate_coefficient(0, n, 'a0n') for n in range(1, N+1)]
    b0n = [calculate_coefficient(0, n, 'b0n') for n in range(1, N+1)]
    am0 = [calculate_coefficient(m, 0, 'am0') for m in range(1, M+1)]
    bm0 = [calculate_coefficient(m, 0, 'bm0') for m in range(1, M+1)]
    
    cmn = [[calculate_coefficient(m, n, 'cmn') for n in range(1, N+1)] for m in range(1, M+1)]
    dmn = [[calculate_coefficient(m, n, 'dmn') for n in range(1, N+1)] for m in range(1, M+1)]
    amn = [[calculate_coefficient(m, n, 'amn') for n in range(1, N+1)] for m in range(1, M+1)]
    bmn = [[calculate_coefficient(m, n, 'bmn') for n in range(1, N+1)] for m in range(1, M+1)]

    
    return a00, a0n, am0, b0n, bm0, cmn, dmn, amn, bmn


def great(t):
    if (carrier(Ac, wc, t) > modulating(Am, wm, t) ):
        return 1.5
    else:
        return 0


h = 0.01
t = np.linspace(0, 1*np.pi, int(1*np.pi/h))
out = [great(_) for _ in t]

# Calculate double Fourier coefficients and reconstruct the wave
T1 = T2 = np.pi  # Assuming the periods are pi
a00, a0n, am0, b0n, bm0, cmn, dmn, amn, bmn, reconstructed_wave, t1_vals, t2_vals = double_fourier(out, t, T1, T2)


plt.plot(t, out)


plt.plot(t1_vals, reconstructed_wave[:, 0], color = 'purple')

plt.show()


