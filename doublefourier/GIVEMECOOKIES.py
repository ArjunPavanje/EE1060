import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, fftfreq
from scipy import signal, integrate

# Define frequencies and amplitude values
w0 = 0.5   # Frequency of the modulating signal
wc = 1.5  # Frequency of the carrier signal

modulation_ratio = 0.8  # Modulation ratio
carrier_amp = 1  # Amplitude of the carrier signal
modulating_amp = modulation_ratio*carrier_amp  # Amplitude of the modulating signal

# Define functions for carrier and modulating signals
def carrier(A, w, t):
    """
    Generates a carrier waveform based on a modified cosine function.
    This function creates a triangular waveform by transforming the cosine function.
    """
    return A*signal.sawtooth(wc*t, 0.5)

def modulating(A, w, t):
    """
    Generates a sinusoidal modulating waveform.
    """
    return A * np.sin(w * t)

def grater(t):
    """
    Compares the modulating signal with the carrier signal to generate a PWM output.
    If the modulating signal is smaller than the carrier signal, output a high value (1.5), otherwise 0.
    """
    mod_signal = modulating(modulating_amp, w0, t)
    carr_signal = carrier(carrier_amp, wc, t)
    return 2 if mod_signal > carr_signal else 0

# Generate time values
t = np.linspace(0, 2*np.pi, 1000)  # Time vector with 1000 points from 0 to 1
pwm = np.array([grater(_) for _ in t])

y_1 = np.linspace(0, 40, 1000)
#a_0_arr = []
#a_1_arr = []
b_1_arr = []


for y in y_1:
    def grater_const(t):
        mod_signal = modulating(modulating_amp, w0, y)
        carr_signal = carrier(carrier_amp, wc, wc*t)
        return 2 if mod_signal > carr_signal else 0

    #a_0, _ = (integrate.quad(grater_const, 0, 2*np.pi, limit = 100, epsabs = 1e-10))
    #a_0 = a_0/(2*np.pi)

    def grater_into_cos(x):
        return grater_const(x)*np.cos(4*x)

    def grater_into_sin(x):
        return grater_const(x)*np.sin(10*x)

    #a_1, _ = (integrate.quad(grater_into_cos, -np.pi, np.pi, limit = 100, epsabs = 1e-10))
    b_1, _ = (integrate.quad(grater_into_sin, -np.pi, np.pi, limit = 100, epsabs = 1e-10))
    #a_1 = a_1/np.pi
    b_1 = b_1/np.pi

    #a_0_arr.append(a_0)
    #a_1_arr.append(a_1)
    b_1_arr.append(b_1)

#plt.plot(y_1, a_0_arr, color = 'r', label = 'a0')
#plt.plot(y_1, a_1_arr, color = 'g', label = 'a4')
#plt.plot(y_1, b_1_arr, color = 'magenta', label = 'b10')
#plt.plot(t, [grater_const(_) for _ in t])

# Plot results
#plt.plot(t, pwm, label="PWM Output")
#plt.plot(t, modulating(modulating_amp, w0, t), label="Modulating Signal")
plt.plot(t, carrier(carrier_amp, wc, t), label="Carrier Signal")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("PWM Generation using Carrier and Modulating Signals")
plt.grid()
plt.show()
