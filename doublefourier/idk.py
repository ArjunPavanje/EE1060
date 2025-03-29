import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

wm = 5  # sin wave's frequency
m = 4  # f_c = m * f_m, m in N
wc = m * wm  # triangular wave's frequency 

Ac = 1  # sin wave's amplitude
mf = 0.1  # Ac = mf * Am, mf < 1
Am = mf * Ac

def carrier(A, w, t):  # triangular wave
    return (A) - (2 * A / np.pi) * np.arccos(np.cos(w * t - (np.pi / 2)))

def modulating(A, w, t):  # sin wave
    return A * np.sin(w * t)

def double_fourier(out, t, T1, T2):
    N = M = 10  # Number of harmonics to consider

    def f(t1, t2):
        idx = int(t1 / (T1 / len(t)))
        if idx >= len(out):
            idx = len(out) - 1
        return out[idx]
    
    # Using simpsonon's rule for integration (integrating over a grid of t1, t2)
    a00 = (1 / (T1 * T2)) * simpson([simpson([f(t1, t2) for t2 in np.linspace(0, T2, 100)], dx=T2 / 100) for t1 in np.linspace(0, T1, 100)], dx=T1 / 100)

    def calculate_coefficient(m, n, coeff_type):
        if coeff_type == 'a0n':
            return (2 / (T1 * T2)) * simpson([f(0, t2) * np.cos(2 * np.pi * n * t2 / T2) for t2 in np.linspace(0, T2, 100)], dx=T2 / 100)
        elif coeff_type == 'b0n':
            return (2 / (T1 * T2)) * simpson([f(0, t2) * np.sin(2 * np.pi * n * t2 / T2) for t2 in np.linspace(0, T2, 100)], dx=T2 / 100)
        elif coeff_type == 'am0':
            return (2 / (T1 * T2)) * simpson([f(t1, 0) * np.cos(2 * np.pi * m * t1 / T1) for t1 in np.linspace(0, T1, 100)], dx=T1 / 100)
        elif coeff_type == 'bm0':
            return (2 / (T1 * T2)) * simpson([f(t1, 0) * np.sin(2 * np.pi * m * t1 / T1) for t1 in np.linspace(0, T1, 100)], dx=T1 / 100)
        elif coeff_type == 'cmn':
            return (4 / (T1 * T2)) * simpson([simpson([f(t1, t2) * np.cos(2 * np.pi * m * t1 / T1) * np.cos(2 * np.pi * n * t2 / T2) for t2 in np.linspace(0, T2, 100)], dx=T2 / 100) for t1 in np.linspace(0, T1, 100)], dx=T1 / 100)
        elif coeff_type == 'dmn':
            return (4 / (T1 * T2)) * simpson([simpson([f(t1, t2) * np.sin(2 * np.pi * m * t1 / T1) * np.sin(2 * np.pi * n * t2 / T2) for t2 in np.linspace(0, T2, 100)], dx=T2 / 100) for t1 in np.linspace(0, T1, 100)], dx=T1 / 100)

    # Calculate Fourier coefficients
    a0n = [calculate_coefficient(0, n, 'a0n') for n in range(1, N + 1)]
    b0n = [calculate_coefficient(0, n, 'b0n') for n in range(1, N + 1)]
    am0 = [calculate_coefficient(m, 0, 'am0') for m in range(1, M + 1)]
    bm0 = [calculate_coefficient(m, 0, 'bm0') for m in range(1, M + 1)]

    cmn = [[calculate_coefficient(m, n, 'cmn') for n in range(1, N + 1)] for m in range(1, M + 1)]
    dmn = [[calculate_coefficient(m, n, 'dmn') for n in range(1, N + 1)] for m in range(1, M + 1)]

    amn = [[0.5 * (cmn[m - 1][n - 1] - dmn[m - 1][n - 1]) for n in range(1, N + 1)] for m in range(1, M + 1)]
    bmn = [[0.5 * (cmn[m - 1][n - 1] + dmn[m - 1][n - 1]) for n in range(1, N + 1)] for m in range(1, M + 1)]

    # Reconstruct the wave using the calculated coefficients
    t1_vals = np.linspace(0, T1, 100)
    t2_vals = np.linspace(0, T2, 100)
    reconstructed_wave = np.zeros((len(t1_vals), len(t2_vals)))

    for i, t1 in enumerate(t1_vals):
        for j, t2 in enumerate(t2_vals):
            sum_terms = a00
            
            # Add a0n and b0n terms
            for n in range(1, N + 1):
                sum_terms += a0n[n - 1] * np.cos(2 * np.pi * n * t2 / T2)
                sum_terms += b0n[n - 1] * np.sin(2 * np.pi * n * t2 / T2)

            # Add am0 and bm0 terms
            for m in range(1, M + 1):
                sum_terms += am0[m - 1] * np.cos(2 * np.pi * m * t1 / T1)
                sum_terms += bm0[m - 1] * np.sin(2 * np.pi * m * t1 / T1)

            # Add amn and bmn terms
            for m in range(1, M + 1):
                for n in range(1, N + 1):
                    sum_terms += amn[m - 1][n - 1] * np.cos(2 * np.pi * m * t1 / T1) * np.cos(2 * np.pi * n * t2 / T2)
                    sum_terms += bmn[m - 1][n - 1] * np.sin(2 * np.pi * m * t1 / T1) * np.sin(2 * np.pi * n * t2 / T2)

            reconstructed_wave[i, j] = sum_terms

    return a00, a0n, am0, b0n, bm0, cmn, dmn, amn, bmn, reconstructed_wave, t1_vals, t2_vals


def great(t):
    if (carrier(Ac, wc, t) > modulating(Am, wm, t)):
        return 1.5
    else:
        return 0


h = 0.01
t = np.linspace(0, 1 * np.pi, int(1 * np.pi / h))
out = [great(_) for _ in t]

# Calculate double Fourier coefficients and reconstruct the wave
T1 = T2 = np.pi  # Assuming the periods are pi
a00, a0n, am0, b0n, bm0, cmn, dmn, amn, bmn, reconstructed_wave, t1_vals, t2_vals = double_fourier(out, t, T1, T2)

# Plot the original signal and the reconstructed signal
plt.plot(t, out, label='Original Signal')
plt.plot(t1_vals, reconstructed_wave[:, 0], color='purple', label='Reconstructed Signal')
plt.legend()
plt.show()

