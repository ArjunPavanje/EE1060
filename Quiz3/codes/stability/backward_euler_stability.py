
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

'''Stability of Backward Euler method for various step sizes'''

# Parameters
N = 10  # Number of periods of square wave to plot
T = 1  # Time period of input square wave
alpha = 0.5  # Duty ratio for input square wave
L = 1  # Inductance (H)
tau = 1  # Time constant L/R (s)
h = 0.01  # Default step size

# Function to generate square wave
def square(t, alpha, T, amplitude=10):
    """Square wave function for numerical methods"""
    return amplitude if (t/T - np.floor(t/T)) < alpha else 0

# Backward Euler implementation
def backward_euler(T, tau, L, alpha, h, t_values):
    """Backward Euler numerical method implementation"""
    R = L/tau  # Calculate R from tau and L
    be = []
    y = 0  # Initial current
    x = 0  # Initial time
    
    for i in range(len(t_values)):
        be.append(y)
        y = ((L*y) + (h*square(x+h, alpha, T)))/(L+h*R)
        x += h
    
    return be

# Create the figure and the line that we will manipulate
fig, ax = plt.subplots(figsize=(12, 8))
plt.subplots_adjust(bottom=0.35)

# Step sizes to compare
step_sizes = [0.01, 0.02, 0.05, 0.1, 1, 2]
current_step_size = step_sizes[0]  # Default step size for display
show_all = False  # Flag to show all step sizes

# Time array
t = np.linspace(0, N*T, int(N*T/h))

# Initial plot
lines = []
for h_val in step_sizes:
    t_val = np.linspace(0, N*T, int(N*T/h_val))
    current = backward_euler(T, tau, L, alpha, h_val, t_val)
    line, = ax.plot(t_val, current, lw=2, label=f'h = {h_val}', visible=False)
    lines.append(line)

# Make first line visible
lines[0].set_visible(True)

ax.set_xlabel('Time [s]')
ax.set_ylabel('Current [A]')
ax.set_title(f'Backward Euler with h = {current_step_size}')
ax.grid(True)
ax.legend()

# Make sliders to control parameters
slider_width = 0.65
slider_height = 0.03
slider_left = 0.25
slider_bottom_start = 0.2

axtime = plt.axes([slider_left, slider_bottom_start, slider_width, slider_height])
axalpha = plt.axes([slider_left, slider_bottom_start - 0.05, slider_width, slider_height])
axtau = plt.axes([slider_left, slider_bottom_start - 0.10, slider_width, slider_height])
axL = plt.axes([slider_left, slider_bottom_start - 0.15, slider_width, slider_height])
axh = plt.axes([slider_left, slider_bottom_start - 0.20, slider_width, slider_height])

time_slider = Slider(
    ax=axtime,
    label='Time Period T',
    valmin=0.1,
    valmax=5,
    valinit=T,
)
alpha_slider = Slider(
    ax=axalpha,
    label='Alpha (Duty Cycle)',
    valmin=0.01,
    valmax=0.99,
    valinit=alpha,
)
tau_slider = Slider(
    ax=axtau,
    label='Time Constant τ = L/R [s]',
    valmin=0.01,
    valmax=5,
    valinit=tau,
)
L_slider = Slider(
    ax=axL,
    label='Inductance L [H]',
    valmin=0.01,
    valmax=5,
    valinit=L,
)
h_slider = Slider(
    ax=axh,
    label='Step size h',
    valmin=0.001,
    valmax=0.1,
    valinit=h,
)

# Add buttons for step size selection
btn_width = 0.1
btn_height = 0.05
btn_left_start = 0.1
btn_bottom = 0.25

button_axes = []
buttons = []

for i, h_val in enumerate(step_sizes):
    button_ax = plt.axes([btn_left_start + i*(btn_width + 0.02), btn_bottom, btn_width, btn_height])
    button_axes.append(button_ax)
    button = Button(button_ax, f'h = {h_val}')
    buttons.append(button)

# Add "Show All" button
all_button_ax = plt.axes([btn_left_start + len(step_sizes)*(btn_width + 0.02), btn_bottom, btn_width, btn_height])
all_button = Button(all_button_ax, 'Show All')

def update(val=None):
    # Get current slider values
    T_val = time_slider.val
    tau_val = tau_slider.val
    L_val = L_slider.val
    alpha_val = alpha_slider.val
    h_val = h_slider.val
    
    # Update plot for each step size
    for i, h_size in enumerate(step_sizes):
        t_val = np.linspace(0, N*T_val, int(N*T_val/h_size))
        current = backward_euler(T_val, tau_val, L_val, alpha_val, h_size, t_val)
        lines[i].set_xdata(t_val)
        lines[i].set_ydata(current)
    
    # Update title if not showing all
    if not show_all:
        ax.set_title(f'Backward Euler with h = {current_step_size}')
    else:
        ax.set_title('Backward Euler - Comparison of Different Step Sizes')
    
    # Adjust plot limits
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw_idle()

def show_step_size(event, h_index):
    global current_step_size, show_all
    current_step_size = step_sizes[h_index]
    show_all = False
    
    # Hide all lines
    for line in lines:
        line.set_visible(False)
    
    # Show only the selected line
    lines[h_index].set_visible(True)
    
    ax.set_title(f'Backward Euler with h = {current_step_size}')
    ax.legend()
    fig.canvas.draw_idle()

def show_all_step_sizes(event):
    global show_all
    show_all = True
    
    # Show all lines
    for line in lines:
        line.set_visible(True)
    
    ax.set_title('Backward Euler - Comparison of Different Step Sizes')
    ax.legend()
    fig.canvas.draw_idle()

# Register callbacks
time_slider.on_changed(update)
alpha_slider.on_changed(update)
tau_slider.on_changed(update)
L_slider.on_changed(update)
h_slider.on_changed(update)

for i, button in enumerate(buttons):
    button.on_clicked(lambda event, idx=i: show_step_size(event, idx))

all_button.on_clicked(show_all_step_sizes)

# Initial update to ensure everything is in sync
update()

plt.show()
