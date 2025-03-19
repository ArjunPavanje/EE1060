import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider, RadioButtons

# Parameters
N = 10  # Number of periods to plot
h = 0.01  # Step Size
alpha = 0.5  # duty ratio for input square wave
tau = 1  # time constant (L/R)
L = 1  # inductance
T = 1  # time period of input square wave

def fourier_current(T, tau, L, alpha, h, n_terms, t_values):
    """Calculate current using Fourier series from equation 0.20 and 0.21"""
    w0 = 2*np.pi/T
    R = L/tau  # Calculate R from tau and L
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

def square_wave(t, T, alpha, amplitude=10):
    """Generate square wave voltage with duty cycle alpha"""
    return amplitude * ((t/T - np.floor(t/T)) < alpha)

# Create time array
t = np.linspace(0, N*T, int(N*T/h))

# Define n values to compare
n_values = [1, 5, 10, 50, 100, 1000]
colors = ['red', 'orange', 'green', 'cyan', 'magenta', 'lawngreen']
labels = [f"n = {n}" for n in n_values]
labels.append("All n values")

# Create figure with more space for controls
fig = plt.figure(figsize=(14, 10))
# Create main plot area
ax = fig.add_axes([0.1, 0.35, 0.8, 0.55])

# Initial plot
lines = []
for i, n_val in enumerate(n_values):
    current = fourier_current(T, tau, L, alpha, h, n_val, t)
    line, = ax.plot(t, current, color=colors[i], lw=2, label=f"n = {n_val}", visible=True)
    lines.append(line)

# Add square wave for reference (scaled)
voltage = square_wave(t, T, alpha)
voltage_line, = ax.plot(t, voltage*0.5, 'k--', alpha=0.5, label="Input Voltage (scaled)")

# Create a separate legend for "All n values" mode
all_n_legend_elements = []
for i, n_val in enumerate(n_values):
    all_n_legend_elements.append(plt.Line2D([0], [0], color=colors[i], lw=2, label=f"n = {n_val}"))
all_n_legend_elements.append(plt.Line2D([0], [0], color='k', linestyle='--', alpha=0.5, label="Input Voltage (scaled)"))
all_n_legend = ax.legend(handles=all_n_legend_elements, loc='upper right', fontsize=10, framealpha=0.9)

# Add labels and title
ax.set_xlabel('Time [s]', fontsize=12)
ax.set_ylabel('Current [A]', fontsize=12)
ax.set_title('Fourier Series Solution with Different Numbers of Terms', fontsize=14)
ax.grid(True)

# Add sliders in a separate area below the plot
slider_width = 0.65
slider_height = 0.03
slider_left = 0.25
slider_bottom_start = 0.22

axtime = fig.add_axes([slider_left, slider_bottom_start, slider_width, slider_height])
axalpha = fig.add_axes([slider_left, slider_bottom_start - 0.04, slider_width, slider_height])
axtau = fig.add_axes([slider_left, slider_bottom_start - 0.08, slider_width, slider_height])
axL = fig.add_axes([slider_left, slider_bottom_start - 0.12, slider_width, slider_height])
axh = fig.add_axes([slider_left, slider_bottom_start - 0.16, slider_width, slider_height])

time_slider = Slider(
    ax=axtime,
    label='Time Period T',
    valmin=0.1,
    valmax=5.0,
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
    valmin=0.1,
    valmax=10.0,
    valinit=tau,
)
L_slider = Slider(
    ax=axL,
    label='Inductance L [H]',
    valmin=0.01,
    valmax=5.0,
    valinit=L,
)
h_slider = Slider(
    ax=axh,
    label='Step size h',
    valmin=0.001,
    valmax=0.1,
    valinit=h,
)

# Add radio buttons for n value selection
rax = fig.add_axes([0.025, 0.4, 0.15, 0.3])
radio = RadioButtons(rax, labels)

# Function to update plot based on slider values
def update(val):
    T_val = time_slider.val
    tau_val = tau_slider.val
    L_val = L_slider.val
    alpha_val = alpha_slider.val
    h_val = h_slider.val
    
    # Update time array if h changed
    global t, voltage, voltage_line
    t = np.linspace(0, N*T_val, int(N*T_val/h_val))
    
    # Update voltage
    voltage = square_wave(t, T_val, alpha_val)
    voltage_line.set_xdata(t)
    voltage_line.set_ydata(voltage*0.5)
    
    # Update currents for all n values
    for i, n_val in enumerate(n_values):
        current = fourier_current(T_val, tau_val, L_val, alpha_val, h_val, n_val, t)
        lines[i].set_xdata(t)
        lines[i].set_ydata(current)
    
    # Update plot limits
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw_idle()

# Function to handle radio button selection
def select_n(label):
    if label == "All n values":
        for line in lines:
            line.set_visible(True)
        # Update legend to show all n values
        ax.legend(handles=all_n_legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
    else:
        n_idx = labels.index(label)
        for i, line in enumerate(lines):
            line.set_visible(i == n_idx)
        # Update legend to show only the selected n value
        legend_elements = [
            plt.Line2D([0], [0], color=colors[n_idx], lw=2, label=f"n = {n_values[n_idx]}"),
            plt.Line2D([0], [0], color='k', linestyle='--', alpha=0.5, label="Input Voltage (scaled)")
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
    
    fig.canvas.draw_idle()

# Register callbacks
time_slider.on_changed(update)
alpha_slider.on_changed(update)
tau_slider.on_changed(update)
L_slider.on_changed(update)
h_slider.on_changed(update)
radio.on_clicked(select_n)

# Create reset button
resetax = fig.add_axes([0.8, 0.25, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')

def reset(event):
    time_slider.reset()
    alpha_slider.reset()
    tau_slider.reset()
    L_slider.reset()
    h_slider.reset()
    
    # Manually set all lines to visible and update legend
    for line in lines:
        line.set_visible(True)
    ax.legend(handles=all_n_legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
    
    # Set radio button to "All n values" without triggering the callback
    radio.set_active(len(n_values))
    
    update(None)
    
button.on_clicked(reset)

# Set initial state to show all n values
radio.set_active(len(n_values))  # This is the index of "All n values"

# Add text explaining the equation
equation_text = "Using equations (0.20) and (0.21) from the image"
fig.text(0.5, 0.02, equation_text, ha='center', fontsize=10)

plt.show()

