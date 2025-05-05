import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from matplotlib.widgets import Slider, Button, RadioButtons

# Define signal and kernel functions
def signal(t, A, w, phi):
    return A * np.sin(w * t + phi)

def box_kernel(t, T, t0):
    return np.where((t > -T - t0) & (t < T - t0), 1, 0)

# Initial parameters
A0, w0, phi0, t0_0, T0 = 1, 1, 0, 0, 2

# Create figure and axis
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.4)

# Initial time vector
t = np.linspace(-30, 30, 2000)

# Initial plot
sig = signal(t, A0, w0, phi0)
kern = box_kernel(t, T0, t0_0)
conv = fftconvolve(sig, kern, mode='same') * (t[1] - t[0])

# Plot elements
line_conv, = ax.plot(t, conv, lw=2, label='Convolution')
line_sig, = ax.plot(t, sig, lw=2, linestyle='--', label='Original Sine')
ax.set_ylim(-5, 5)
ax.legend()

# Create sliders
ax_A = plt.axes([0.25, 0.3, 0.65, 0.03])
ax_w = plt.axes([0.25, 0.25, 0.65, 0.03])
ax_phi = plt.axes([0.25, 0.2, 0.65, 0.03])
ax_t0 = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_T = plt.axes([0.25, 0.1, 0.65, 0.03])

s_A = Slider(ax_A, 'A', 0.1, 5.0, valinit=A0)
s_w = Slider(ax_w, 'ω', 0.1, 10.0, valinit=w0)
s_phi = Slider(ax_phi, 'φ', -np.pi, np.pi, valinit=phi0)
s_t0 = Slider(ax_t0, 't₀', -15.0, 15.0, valinit=t0_0)
s_T = Slider(ax_T, 'T', 0.1, 10.0, valinit=T0)

# Create radio buttons
rax = plt.axes([0.05, 0.5, 0.1, 0.2])
radio = RadioButtons(rax, ('Standard', 'Shifted', 't>0'), active=0)
plt.setp(radio.labels, fontsize=10)

# Create reset button
reset_ax = plt.axes([0.05, 0.4, 0.1, 0.04])
reset_btn = Button(reset_ax, 'Reset')

# Create compare button
compare_ax = plt.axes([0.05, 0.35, 0.1, 0.04])
compare_btn = Button(compare_ax, 'Compare All')

# Track convolution type
current_case = 'standard'
compare_mode = [False]  # Use list to allow modification in nested functions

def update(val):
    if compare_mode[0]:
        return  # Don't update single plots if in compare mode
    A = s_A.val
    w = s_w.val
    phi = s_phi.val
    t0 = s_t0.val
    T = s_T.val
    
    sig = signal(t, A, w, phi)
    
    # Update kernel and compute convolution based on current case
    if current_case == 'standard':
        kern = box_kernel(t, T, 0)
        conv = fftconvolve(sig, kern, mode='same') * (t[1] - t[0])
        xlim_left = min(-10, -T) - 2
        xlim_right = max(10, T) + 2
    elif current_case == 'shifted':
        kern = box_kernel(t, T, t0)
        conv = fftconvolve(sig, kern, mode='same') * (t[1] - t[0])
        xlim_left = min(-10, t0 - T) - 2
        xlim_right = max(10, t0 + T) + 2
    elif current_case == 't>0' or current_case == 'tpos':
        conv = (A/w) * (np.cos(w*(t - T) + phi) - np.cos(w*t + phi))
        xlim_left = min(-10, 0) - 2
        xlim_right = max(10, T) + 2
    
    # Update plots and axes
    line_conv.set_ydata(conv)
    line_sig.set_ydata(sig)
    ax.set_xlim(xlim_left, xlim_right)
    ax.set_ylim(-1.5*(A+T), 1.5*(A+T))
    ax.legend()
    fig.canvas.draw_idle()

# Connect sliders
s_A.on_changed(update)
s_w.on_changed(update)
s_phi.on_changed(update)
s_t0.on_changed(update)
s_T.on_changed(update)

# Radio button handler
def mode_handler(label):
    global current_case
    if label.lower() == 't>0':
        current_case = 't>0'
    else:
        current_case = label.lower()
    compare_mode[0] = False
    # Restore only the two main lines
    ax.clear()
    ax.set_ylim(-5, 5)
    line_conv, = ax.plot(t, np.zeros_like(t), lw=2, label='Convolution')
    line_sig, = ax.plot(t, np.zeros_like(t), lw=2, linestyle='--', label='Original Sine')
    ax.legend()
    # Update lines globally
    globals()['line_conv'] = line_conv
    globals()['line_sig'] = line_sig
    update(None)

radio.on_clicked(mode_handler)

# Reset handler
def reset(event):
    s_A.reset()
    s_w.reset()
    s_phi.reset()
    s_t0.reset()
    s_T.reset()
    radio.set_active(0)
    global current_case
    current_case = 'standard'
    compare_mode[0] = False
    # Restore only the two main lines
    ax.clear()
    ax.set_ylim(-5, 5)
    line_conv, = ax.plot(t, np.zeros_like(t), lw=2, label='Convolution')
    line_sig, = ax.plot(t, np.zeros_like(t), lw=2, linestyle='--', label='Original Sine')
    ax.legend()
    # Update lines globally
    globals()['line_conv'] = line_conv
    globals()['line_sig'] = line_sig
    update(None)

reset_btn.on_clicked(reset)

# Compare all handler
def compare_all(event):
    compare_mode[0] = True
    A = s_A.val
    w = s_w.val
    phi = s_phi.val
    t0 = s_t0.val
    T = s_T.val
    
    sig = signal(t, A, w, phi)
    conv_standard = fftconvolve(sig, box_kernel(t, T, 0), mode='same') * (t[1] - t[0])
    conv_shifted = fftconvolve(sig, box_kernel(t, T, t0), mode='same') * (t[1] - t[0])
    conv_tpos = (A/w) * (np.cos(w*(t - T) + phi) - np.cos(w*t + phi))
    
    ax.clear()
    ax.plot(t, conv_standard, lw=2, label='Standard')
    ax.plot(t, conv_shifted, lw=2, label='Shifted')
    ax.plot(t, conv_tpos, lw=2, label='t>0')
    ax.plot(t, sig, lw=2, linestyle='--', label='Original Sine')
    ax.set_ylim(-1.5*(A+T), 1.5*(A+T))
    ax.set_xlim(-32, 32)
    ax.legend()
    ax.set_title('Comparison of Convolution Outputs')
    fig.canvas.draw_idle()

compare_btn.on_clicked(compare_all)

plt.show()
