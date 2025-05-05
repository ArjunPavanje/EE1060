
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

# Define exponential signal
def signal(t, a):
    return np.exp(a * t)

# Initial parameters
a0, t0_0, T0 = 1.0, 0.0, 2.0

# Create figure and axis
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.4)

# Initial time vector
t = np.linspace(-10, 10, 10000)

# Initial plot
sig = signal(t, a0)
conv_standard = (2 * np.exp(a0 * t) / a0) * np.sinh(a0 * T0)

# Plot elements
line_conv, = ax.plot(t, conv_standard, lw=2, label='Convolution')
line_sig, = ax.plot(t, sig, lw=2, linestyle='--', label=r'$e^{at}$')
ax.set_ylim(-5, 50)
ax.legend()
ax.set_xlabel('Time')
ax.set_title('Exponential Signal Convolution')

# Create sliders
ax_a = plt.axes([0.25, 0.3, 0.65, 0.03])
ax_T = plt.axes([0.25, 0.25, 0.65, 0.03])
ax_t0 = plt.axes([0.25, 0.2, 0.65, 0.03])

s_a = Slider(ax_a, 'a', 0.1, 5.0, valinit=a0)
s_T = Slider(ax_T, 'T', 0.1, 5.0, valinit=T0)
s_t0 = Slider(ax_t0, 't₀', -5.0, 5.0, valinit=t0_0)

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

# Track convolution type and compare mode
current_case = 'standard'
compare_mode = [False]  # Use list to make it mutable within nested functions

def update(val):
    if compare_mode[0]:
        return  # Don't update if in compare mode
        
    a = s_a.val
    T = s_T.val
    t0 = s_t0.val
    
    sig = signal(t, a)
    
    # Calculate convolution based on current case
    if current_case == 'standard':
        conv = (2 * np.exp(a * t) / a) * np.sinh(a * T)
        xlim = [-10, 10]
    elif current_case == 'shifted':
        conv = (2 * np.exp(a * (t - t0)) / a) * np.sinh(a * T)
        xlim = [t0-10, t0+10]
    elif current_case == 't>0':
        conv = (np.exp(a * t) / a) * (1 - np.exp(-a * T))
        xlim = [-2, 10]  # Focus on positive time
    
    # Update plots
    line_conv.set_ydata(conv)
    line_sig.set_ydata(sig)
    ax.set_xlim(xlim)
    ax.set_ylim(-0.5, 1.5*np.max(conv))
    fig.canvas.draw_idle()

# Connect sliders
s_a.on_changed(update)
s_T.on_changed(update)
s_t0.on_changed(update)

# Radio button handler
def mode_handler(label):
    global current_case
    current_case = label.lower()
    compare_mode[0] = False
    # Restore only the two main lines
    ax.clear()
    global line_conv, line_sig
    line_conv, = ax.plot(t, np.zeros_like(t), lw=2, label='Convolution')
    line_sig, = ax.plot(t, np.zeros_like(t), lw=2, linestyle='--', label=r'$e^{at}$')
    ax.legend()
    ax.set_xlabel('Time')
    update(None)

radio.on_clicked(mode_handler)

# Reset handler
def reset(event):
    s_a.reset()
    s_T.reset()
    s_t0.reset()
    radio.set_active(0)
    global current_case
    current_case = 'standard'
    compare_mode[0] = False
    # Restore only the two main lines
    ax.clear()
    global line_conv, line_sig
    line_conv, = ax.plot(t, np.zeros_like(t), lw=2, label='Convolution')
    line_sig, = ax.plot(t, np.zeros_like(t), lw=2, linestyle='--', label=r'$e^{at}$')
    ax.legend()
    ax.set_xlabel('Time')
    update(None)

reset_btn.on_clicked(reset)

# Compare all handler
def compare_all(event):
    compare_mode[0] = True
    a = s_a.val
    T = s_T.val
    t0 = s_t0.val
    
    sig = signal(t, a)
    
    # Calculate all three types of convolutions
    conv_standard = (2 * np.exp(a * t) / a) * np.sinh(a * T)
    conv_shifted = (2 * np.exp(a * (t - t0)) / a) * np.sinh(a * T)
    conv_tpos = (np.exp(a * t) / a) * (1 - np.exp(-a * T))
    
    # Clear the plot and draw all convolutions
    ax.clear()
    ax.plot(t, conv_standard, lw=2, label='Standard')
    ax.plot(t, conv_shifted, lw=2, label='Shifted')
    ax.plot(t, conv_tpos, lw=2, label='t>0')
    ax.plot(t, sig, lw=2, linestyle='--', label=r'$e^{at}$')
    
    # Set appropriate axis limits to show all convolutions
    min_x = min(-10, t0-10, -2)
    max_x = max(10, t0+10, 10)
    ax.set_xlim(min_x, max_x)
    
    max_y = max(np.max(conv_standard), np.max(conv_shifted), np.max(conv_tpos))
    ax.set_ylim(-0.5, 1.5*max_y)
    
    ax.legend()
    ax.set_xlabel('Time')
    ax.set_title('Comparison of All Convolution Types')
    fig.canvas.draw_idle()

compare_btn.on_clicked(compare_all)

update(None)  # Initial update
plt.show()
