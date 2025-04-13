import matplotlib.pyplot as plt
import numpy as np 

wm = 5 # sin wave's frequency
m = 4 # f_c = m f_m, m in n
wc = m * wm # triangular wave's frequency 

ac = 1 # sin wave's amplitude
mf = 0.1 # ac = mf am, mf < 1
am = mf * ac 

def carrier(a, w, t): # triangular wave
    return (a)-(2*a/(np.pi)*np.arccos(np.cos(w*t -(np.pi/2))))

def modulating(a, w, t): # sin wave
    return a*np.sin(w*t)

def great(t):
    if (carrier(ac, wc, t) > modulating(am, wm, t) ):
        return 1.5
    else:
        return 0

h = 0.01
t = np.linspace(0, 9*np.pi, int(9*np.pi/h))
out = [great(_) for _ in t]

#y = np.linspace(-1.1, 1.6, 100)
#x = np.linspace(0, 0, 100)

'''
maybe_period = np.linspace(0, 5*np.pi, int(5*np.pi/h))
for i in maybe_period:
    isperiod = 1
    for x in range(0, int(5*np.pi/h)):
        if abs(out[x] - out[x + int(i*h)]) < 1e-4:
            isperiod = 0
            break
    if (isperiod == 1):
        print("Period is ", i)
'''

# Function to calculate the period by checking for repeating values within a tolerance
def find_period(signal, threshold=1e-4):
    for period in range(0, int(2*np.pi/(5*h))+1):  # Loop through possible period lengths
        is_periodic = True
        '''if period == int(2*np.pi/h):
                print("Value period I will be analysing, ", period)'''
        for i in range(0, int(2*np.pi/(5*h))):  # Check if the waveform repeats within tolerance
            
            '''if i < 10 and period == int(2*np.pi/h):
                print("out[i] = ", signal[i], " out[period + i] = ", signal[period + i])'''
            if abs(signal[i] - signal[i + period]) > threshold and signal[i] == signal [i+1]:
                '''if(period == int(2*np.pi/h)):
                    print("Value of i it breaks at is, ", i)'''
                is_periodic = False
                break
        if is_periodic:
            print("Period is: ", period/100)
            #return period
    return None  # If no period found

find_period(out)

plt.figure()
plt.plot(t, out, label = 'PWM output')
plt.plot(t, modulating(Am, wm, t), label = 'modulating signal')
plt.plot(t, carrier(Ac, wc, t), label = 'carrier signal')
#plt.plot(x, y, color = 'red')
plt.legend()
plt.grid(True)
plt.show()
