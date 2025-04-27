
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

def update(val):
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
    update(None)

# Connect controls
s_T.on_changed(update)
s_t0.on_changed(update)
radio.on_clicked(update)
reset_btn.on_clicked(reset)

update(None)  # Initial update
plt.show()
