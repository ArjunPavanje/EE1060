
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

# Box kernel definition
def h(t, T, t0=0):
    """Box kernel with adjustable position and width"""
    return np.where((t >= t0 - T) & (t <= t0 + T), 1, 0)

# Initial parameters
T0 = 1.0  # Initial width
t0_0 = 0.0  # Initial position

# Create time axis
t = np.linspace(-5, 5, 1000)

# Set up plot
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.35)
line, = ax.plot(t, h(t, T0), lw=2)
ax.set_ylim(-0.1, 1.1)
ax.set_title("Box Kernel Visualization")
ax.grid(True)

# Create sliders
ax_T = plt.axes([0.25, 0.25, 0.65, 0.03])
ax_t0 = plt.axes([0.25, 0.2, 0.65, 0.03])
s_T = Slider(ax_T, 'T', 0.1, 2.0, valinit=T0)
s_t0 = Slider(ax_t0, 't₀', -2.0, 2.0, valinit=t0_0)

# Create radio buttons
rax = plt.axes([0.05, 0.6, 0.1, 0.2])
radio = RadioButtons(rax, ('Standard', 'Shifted', 't>0'), active=0)

# Create reset button
reset_ax = plt.axes([0.05, 0.4, 0.1, 0.04])
reset_btn = Button(reset_ax, 'Reset')

# Create compare button
compare_ax = plt.axes([0.05, 0.35, 0.1, 0.04])
compare_btn = Button(compare_ax, 'Compare All')

# Variable to track compare mode
compare_mode = [False]  # Use list to make it mutable within nested functions

def update(val):
    if compare_mode[0]:
        return  # Don't update if in compare mode
        
    T = s_T.val
    t0 = s_t0.val
    case = radio.value_selected.lower()
    
    if case == 'standard':
        kernel = h(t, T)
        ax.set_xlim(-T-1, T+1)
        ax.set_title("Standard Kernel: h(t)")
    elif case == 'shifted':
        kernel = h(t, T, t0)
        ax.set_xlim(t0-T-1, t0+T+1)
        ax.set_title(f"Shifted Kernel: h(t-t₀), t₀={t0:.1f}")
    elif case == 't>0':
        kernel = np.where((t >= 0) & (t <= T), 1, 0)
        ax.set_xlim(-1, T+1)
        ax.set_title(f"Right-sided Kernel: [0, T] (T={T:.1f})")
    
    line.set_ydata(kernel)
    fig.canvas.draw_idle()

def reset(event):
    s_T.reset()
    s_t0.reset()
    radio.set_active(0)
    compare_mode[0] = False
    ax.clear()
    global line
    line, = ax.plot(t, h(t, T0), lw=2)
    ax.grid(True)
    ax.set_ylim(-0.1, 1.1)
    update(None)

def compare_all(event):
    compare_mode[0] = True
    T = s_T.val
    t0 = s_t0.val
    
    # Calculate all three kernel types
    standard_kernel = h(t, T)
    shifted_kernel = h(t, T, t0)
    tpos_kernel = np.where((t >= 0) & (t <= T), 1, 0)
    
    # Clear plot and redraw all kernels
    ax.clear()
    ax.plot(t, standard_kernel, lw=2, label='Standard')
    ax.plot(t, shifted_kernel, lw=2, label='Shifted')
    ax.plot(t, tpos_kernel, lw=2, label='t>0')
    
    # Set limits to show all kernels
    min_x = min(-T-1, t0-T-1, -1)
    max_x = max(T+1, t0+T+1, T+1)
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True)
    ax.legend()
    ax.set_title("Comparison of All Kernel Types")
    fig.canvas.draw_idle()

def radio_update(label):
    compare_mode[0] = False
    ax.clear()
    global line
    line, = ax.plot(t, np.zeros_like(t), lw=2)
    ax.grid(True)
    ax.set_ylim(-0.1, 1.1)
    update(None)  # Call update to set the correct kernel

# Connect controls
s_T.on_changed(update)
s_t0.on_changed(update)
radio.on_clicked(radio_update)  # Use custom handler
reset_btn.on_clicked(reset)
compare_btn.on_clicked(compare_all)  # Connect compare button

update(None)  # Initial update
plt.show()
