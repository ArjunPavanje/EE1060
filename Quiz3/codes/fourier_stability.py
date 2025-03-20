
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

# Parameters
N = 10  # Number of periods of square wave to plot
T = 1  # Time period of input square wave
alpha = 0.5  # Duty ratio for input square wave
L = 1  # Inductance (H)
tau = 1  # Time constant L/R (s)
h = 0.01  # Default step size
n_terms = 20  # Default number of Fourier terms

# Function to generate square wave
def square(t, alpha, T, amplitude=10):
    """Square wave function for numerical methods"""
    return amplitude if (t/T - np.floor(t/T)) < alpha else 0

# Fourier Series implementation
def fourier_current(T, tau, L, alpha, h, n_terms, t_values):
    """Calculate current using Fourier series from equation 0.20 and 0.21"""
    R = L/tau  # Calculate R from tau and L
    w0 = 2*np.pi/T
    result = []
    
    for t in t_values:
        # First term (DC component)
        current = (10*alpha/R) * (1 - np.exp(-t/tau))
        
        # Sum for the sin(2παn) terms (equation 0.20)
        for n in range(1, int(n_terms+1)):
            term1 = (10/(n*np.pi)) * np.sin(2*np.pi*alpha*n)
            term2 = (R*np.cos(n*w0*t) + n*w0*L*np.sin(n*w0*t))/(R**2 + L**2*(n*w0)**2)
            term3 = -R/(R**2 + L**2*(n*w0)**2) * np.exp(-t/tau)
            current += term1 * (term2 + term3)
        
        # Sum for the (1-cos(2παn)) terms (equation 0.21)
        for n in range(1, int(n_terms+1)):
            term1 = (10/(n*np.pi)) * (1 - np.cos(2*np.pi*alpha*n))
            term2 = (R*np.sin(n*w0*t) - L*n*w0*np.cos(n*w0*t))/(R**2 + L**2*(n*w0)**2)
            term3 = (n*w0*L)/(R**2 + L**2*(n*w0)**2) * np.exp(-t/tau)
            current += term1 * (term2 + term3)
        
        result.append(current)
    
    return result

# Create the figure and the line that we will manipulate
fig, ax = plt.subplots(figsize=(12, 8))
plt.subplots_adjust(bottom=0.4)

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
    current = fourier_current(T, tau, L, alpha, h_val, n_terms, t_val)
    line, = ax.plot(t_val, current, lw=2, label=f'h = {h_val}', visible=False)
    lines.append(line)

# Make first line visible
lines[0].set_visible(True)

ax.set_xlabel('Time [s]')
ax.set_ylabel('Current [A]')
ax.set_title(f'Fourier Series with h = {current_step_size}')
ax.grid(True)
ax.legend()

# Make sliders to control parameters
slider_width = 0.65
slider_height = 0.03
slider_left = 0.25
slider_bottom_start = 0.2

axtime = plt.axes([slider_left, slider_bottom_start, slider_width, slider_height])
axalpha = plt.axes([slider_left, slider_bottom_start - 0.25, slider_width, slider_height])
axtau = plt.axes([slider_left, slider_bottom_start - 0.10, slider_width, slider_height])
axL = plt.axes([slider_left, slider_bottom_start - 0.15, slider_width, slider_height])
axh = plt.axes([slider_left, slider_bottom_start - 0.20, slider_width, slider_height])
axn = plt.axes([slider_left, slider_bottom_start - 0.05, slider_width, slider_height])

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
    valmin=0.005,
    valmax=1,
    valinit=h,
)
n_slider = Slider(
    ax=axn,
    label='Number of Fourier terms',
    valmin=1,
    valmax=100,
    valinit=n_terms,
    valstep=1
)

# Add buttons for step size selection
btn_width = 0.1
btn_height = 0.05
btn_left_start = 0.1
btn_bottom = 0.3

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
    n_val = n_slider.val
    
    # Update plot for each step size
    for i, h_size in enumerate(step_sizes):
        t_val = np.linspace(0, N*T_val, int(N*T_val/h_size))
        current = fourier_current(T_val, tau_val, L_val, alpha_val, h_size, n_val, t_val)
        lines[i].set_xdata(t_val)
        lines[i].set_ydata(current)
    
    # Update title if not showing all
    if not show_all:
        ax.set_title(f'Fourier Series with h = {current_step_size}')
    else:
        ax.set_title('Fourier Series - Comparison of Different Step Sizes')
    
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
    
    ax.set_title(f'Fourier Series with h = {current_step_size}')
    ax.legend()
    fig.canvas.draw_idle()

def show_all_step_sizes(event):
    global show_all
    show_all = True
    
    # Show all lines
    for line in lines:
        line.set_visible(True)
    
    ax.set_title('Fourier Series - Comparison of Different Step Sizes')
    ax.legend()
    fig.canvas.draw_idle()

# Register callbacks
time_slider.on_changed(update)
alpha_slider.on_changed(update)
tau_slider.on_changed(update)
L_slider.on_changed(update)
h_slider.on_changed(update)
n_slider.on_changed(update)

for i, button in enumerate(buttons):
    button.on_clicked(lambda event, idx=i: show_step_size(event, idx))

all_button.on_clicked(show_all_step_sizes)

# Initial update to ensure everything is in sync
update()

plt.show()
