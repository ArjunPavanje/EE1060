import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider, RadioButtons

# Parameters
N = 10  # Number of periods to plot
h = 0.01  # Step Size
alpha = 0.5  # duty ratio for input square wave
T = 1  # time period of input square wave
A = 10  # Amplitude of square wave
custom_n_terms = 20  # Initial value for custom number of terms

def fourier_square_wave(T, alpha, A, n_terms, t_values):
    """Calculate square wave using Fourier series from the image"""
    w0 = 2*np.pi/T
    result = []
    
    for t in t_values:
        # First term (DC component)
        voltage = A * alpha  # a0 term
        
        # Sum for the cosine and sine terms
        for n in range(1, int(n_terms+1)):
            # an coefficient
            an = (A/(n*np.pi)) * np.sin(2*np.pi*alpha*n)
            # bn coefficient
            bn = (A/(n*np.pi)) * (1 - np.cos(2*np.pi*alpha*n))
            
            # Add the terms to the sum
            voltage += an * np.cos(n*w0*t) + bn * np.sin(n*w0*t)
        
        result.append(voltage)
    
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
labels.append("Custom n")

# Create figure with more space for controls
fig = plt.figure(figsize=(14, 10))
# Create main plot area
ax = fig.add_axes([0.1, 0.35, 0.8, 0.55])

# Initial plot
lines = []
for i, n_val in enumerate(n_values):
    voltage_fourier = fourier_square_wave(T, alpha, A, n_val, t)
    line, = ax.plot(t, voltage_fourier, color=colors[i], lw=2, label=f"n = {n_val}", visible=True)
    lines.append(line)

# Add a line for custom n terms
custom_voltage = fourier_square_wave(T, alpha, A, custom_n_terms, t)
custom_line, = ax.plot(t, custom_voltage, color='blue', lw=2, label=f"n = {custom_n_terms}", visible=False)
lines.append(custom_line)  # Add to lines list for visibility control

# Add exact square wave for reference
voltage_exact = square_wave(t, T, alpha, A)
voltage_line, = ax.plot(t, voltage_exact, 'k--', alpha=0.5, label="Exact Square Wave")

# Create a separate legend for "All n values" mode
all_n_legend_elements = []
for i, n_val in enumerate(n_values):
    all_n_legend_elements.append(plt.Line2D([0], [0], color=colors[i], lw=2, label=f"n = {n_val}"))
all_n_legend_elements.append(plt.Line2D([0], [0], color='k', linestyle='--', alpha=0.5, label="Exact Square Wave"))
all_n_legend = ax.legend(handles=all_n_legend_elements, loc='upper right', fontsize=10, framealpha=0.9)

# Add labels and title
ax.set_xlabel('Time [s]', fontsize=12)
ax.set_ylabel('Voltage [V]', fontsize=12)
ax.set_title('Fourier Series Representation of Square Wave', fontsize=14)
ax.grid(True)

# Add sliders in a separate area below the plot
slider_width = 0.65
slider_height = 0.03
slider_left = 0.25
slider_bottom_start = 0.22

axtime = fig.add_axes([slider_left, slider_bottom_start, slider_width, slider_height])
axalpha = fig.add_axes([slider_left, slider_bottom_start - 0.04, slider_width, slider_height])
axamp = fig.add_axes([slider_left, slider_bottom_start - 0.08, slider_width, slider_height])
axh = fig.add_axes([slider_left, slider_bottom_start - 0.12, slider_width, slider_height])
axn_terms = fig.add_axes([slider_left, slider_bottom_start - 0.16, slider_width, slider_height])

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
amp_slider = Slider(
    ax=axamp,
    label='Amplitude A [V]',
    valmin=1.0,
    valmax=20.0,
    valinit=A,
)
h_slider = Slider(
    ax=axh,
    label='Step size h',
    valmin=0.001,
    valmax=0.1,
    valinit=h,
)
n_terms_slider = Slider(
    ax=axn_terms,
    label='Custom n terms',
    valmin=1,
    valmax=5000,
    valinit=custom_n_terms,
    valstep=1
)

# Add radio buttons for n value selection
rax = fig.add_axes([0.025, 0.4, 0.15, 0.3])
radio = RadioButtons(rax, labels)

# Function to update plot based on slider values
def update(val):
    T_val = time_slider.val
    alpha_val = alpha_slider.val
    A_val = amp_slider.val
    h_val = h_slider.val
    
    # Update time array if h changed
    global t, voltage_exact, voltage_line
    t = np.linspace(0, N*T_val, int(N*T_val/h_val))
    
    # Update exact square wave
    voltage_exact = square_wave(t, T_val, alpha_val, A_val)
    voltage_line.set_xdata(t)
    voltage_line.set_ydata(voltage_exact)
    
    # Update Fourier series approximations for all n values
    for i, n_val in enumerate(n_values):
        voltage_fourier = fourier_square_wave(T_val, alpha_val, A_val, n_val, t)
        lines[i].set_xdata(t)
        lines[i].set_ydata(voltage_fourier)
    
    # Update custom n terms line
    custom_n = n_terms_slider.val
    custom_voltage = fourier_square_wave(T_val, alpha_val, A_val, custom_n, t)
    custom_line.set_xdata(t)
    custom_line.set_ydata(custom_voltage)
    custom_line.set_label(f"n = {custom_n}")
    
    # Update plot limits
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw_idle()

# Function to handle radio button selection
def select_n(label):
    if label == "All n values":
        for i, line in enumerate(lines[:-1]):  # All except custom line
            line.set_visible(True)
        custom_line.set_visible(False)
        # Update legend to show all n values
        ax.legend(handles=all_n_legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
    elif label == "Custom n":
        for i, line in enumerate(lines[:-1]):  # All except custom line
            line.set_visible(False)
        custom_line.set_visible(True)
        # Update legend to show only custom n value
        custom_n = int(n_terms_slider.val)
        legend_elements = [
            plt.Line2D([0], [0], color='blue', lw=2, label=f"n = {custom_n}"),
            plt.Line2D([0], [0], color='k', linestyle='--', alpha=0.5, label="Exact Square Wave")
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
    else:
        n_idx = labels.index(label)
        for i, line in enumerate(lines):
            line.set_visible(i == n_idx)
        # Update legend to show only the selected n value
        legend_elements = [
            plt.Line2D([0], [0], color=colors[n_idx], lw=2, label=f"n = {n_values[n_idx]}"),
            plt.Line2D([0], [0], color='k', linestyle='--', alpha=0.5, label="Exact Square Wave")
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
    
    fig.canvas.draw_idle()

# Function to update only the custom n terms plot
def update_custom_n(val=None):
    custom_n = int(n_terms_slider.val)
    T_val = time_slider.val
    alpha_val = alpha_slider.val
    A_val = amp_slider.val
    
    custom_voltage = fourier_square_wave(T_val, alpha_val, A_val, custom_n, t)
    custom_line.set_ydata(custom_voltage)
    custom_line.set_label(f"n = {custom_n}")
    
    # If custom n is selected, update the legend
    if radio.value_selected == "Custom n":
        legend_elements = [
            plt.Line2D([0], [0], color='blue', lw=2, label=f"n = {custom_n}"),
            plt.Line2D([0], [0], color='k', linestyle='--', alpha=0.5, label="Exact Square Wave")
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
    
    fig.canvas.draw_idle()

# Register callbacks
time_slider.on_changed(update)
alpha_slider.on_changed(update)
amp_slider.on_changed(update)
h_slider.on_changed(update)
n_terms_slider.on_changed(update_custom_n)
radio.on_clicked(select_n)

# Create reset button
resetax = fig.add_axes([0.8, 0.25, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')

# Create update custom n button
update_n_ax = fig.add_axes([0.8, 0.20, 0.1, 0.04])
update_n_button = Button(update_n_ax, 'Update n', hovercolor='0.975')
update_n_button.on_clicked(update_custom_n)

def reset(event):
    time_slider.reset()
    alpha_slider.reset()
    amp_slider.reset()
    h_slider.reset()
    n_terms_slider.reset()
    
    # Manually set all lines to visible and update legend
    for i, line in enumerate(lines[:-1]):  # All except custom line
        line.set_visible(True)
    custom_line.set_visible(False)
    ax.legend(handles=all_n_legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
    
    # Set radio button to "All n values" without triggering the callback
    radio.set_active(len(n_values))
    
    update(None)
    
button.on_clicked(reset)

# Set initial state to show all n values
radio.set_active(len(n_values))  # This is the index of "All n values"

# Add text explaining the equation
equation_text = "v(t) = a₀ + Σ[aₙcos(2πnt/T) + bₙsin(2πnt/T)]"
fig.text(0.5, 0.02, equation_text, ha='center', fontsize=12)

plt.show()
