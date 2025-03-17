import matplotlib.pyplot as plt
import numpy as np
import math 
from matplotlib.widgets import Button, Slider
from matplotlib.widgets import RangeSlider

N = 10 # Number of periods to plot
h = 0.01 # Step Size
n = 100

def fourier_current(T, R, L, alpha, h, n_terms, t_values):
    """Calculate current using Fourier series from equation 0.20 and 0.21"""
    tau = L/R
    w0 = 2*np.pi/T
    result = []
    
    for t in t_values:
        # First term (DC component)
        current = (10*alpha/R) * (1 - np.exp(-R*t/L))
        
        # Sum for the sin(2παn) terms (equation 0.20)
        for n in range(1, int(n_terms+1)):
            term1 = (10/(n*np.pi)) * np.sin(2*np.pi*alpha*n)
            term2 = (R*np.cos(n*w0*t) + n*w0*L*np.sin(n*w0*t))/(R**2 + L**2*(n*w0)**2)
            term3 = -R/(R**2 + L**2*(n*w0)**2) * np.exp(-R*t/L)
            current += term1 * (term2 + term3)
        
        # Sum for the (1-cos(2παn)) terms (equation 0.21)
        for n in range(1, int(n_terms+1)):
            term1 = (10/(n*np.pi)) * (1 - np.cos(2*np.pi*alpha*n))
            term2 = (R*np.sin(n*w0*t) - L*n*w0*np.cos(n*w0*t))/(R**2 + L**2*(n*w0)**2)
            term3 = (n*w0*L)/(R**2 + L**2*(n*w0)**2) * np.exp(-R*t/L)
            current += term1 * (term2 + term3)
        
        result.append(current)
    
    return result

def square_wave(t, T, alpha, amplitude=10):
    """Generate square wave voltage with duty cycle alpha"""
    return amplitude * ((t/T - np.floor(t/T)) < alpha)

def plot_type_update(val):
    global plot_type
    plot_type = val
    update(None)

alpha = 0.5 # duty ratio for input square wave
R = 1 # resistance
L = 1 # inductance
T = 1 # time period of input square wave
plot_type = "Current" # Default plot type

t = np.linspace(0, N*T, int(N*T/h))

# Create the figure and the line that we will manipulate
fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot(t, fourier_current(T, R, L, alpha, h, n, t), lw=2, label="Current I(t)")
ax.set_xlabel('Time [s]')
ax.set_ylabel('Current [A]')
ax.set_title('RL Circuit Response')

# adjust the main plot to make room for the sliders
fig.subplots_adjust(left=0.25, bottom=0.4)

# Make sliders to control parameters
axtime = fig.add_axes([0.25, 0.1, 0.65, 0.03])
axalpha = fig.add_axes([0.25, 0.15, 0.65, 0.03])
axR = fig.add_axes([0.25, 0.20, 0.65, 0.03])
axL = fig.add_axes([0.25, 0.25, 0.65, 0.03])
axh = fig.add_axes([0.25, 0.30, 0.65, 0.03])
axn = fig.add_axes([0.25, 0.35, 0.65, 0.03])

time_slider = Slider(
    ax=axtime,
    label='Time Period',
    valmin=0.1,
    valmax=30,
    valinit=T,
)
alpha_slider = Slider(
    ax=axalpha,
    label='Alpha (Duty Cycle)',
    valmin=0.01,
    valmax=0.99,
    valinit=alpha,
)
R_slider = Slider(
    ax=axR,
    label='Resistance R [Ω]',
    valmin=0.1,
    valmax=100,
    valinit=R,
    valstep=0.1
)
L_slider = Slider(
    ax=axL,
    label='Inductance L [H]',
    valmin=0.01,
    valmax=10,
    valinit=L,
    valstep=0.01
)
h_slider = Slider(
    ax=axh,
    label='Step size h',
    valmin=1e-3,
    valmax=1e-1,
    valinit=h,
)
n_slider = Slider(
    ax=axn,
    label='Number of Fourier terms',
    valmin=1,
    valmax=100,
    valinit=20,
    valstep=1
)

# Add radio buttons for plot type selection
rax = fig.add_axes([0.025, 0.5, 0.15, 0.15])
radio = plt.matplotlib.widgets.RadioButtons(
    rax, ('Current', 'Voltage', 'Both'), active=0)

def update(val):
    T_val = time_slider.val
    R_val = R_slider.val
    L_val = L_slider.val
    alpha_val = alpha_slider.val
    h_val = h_slider.val
    n_val = n_slider.val
    
    # Update time array if h changed
    global t
    t = np.linspace(0, N*T_val, int(N*T_val/h_val))
    
    # Clear previous plot
    ax.clear()
    
    if plot_type == "Current":
        current = fourier_current(T_val, R_val, L_val, alpha_val, h_val, n_val, t)
        line, = ax.plot(t, current, lw=2, label="Current I(t)")
        ax.set_ylabel('Current [A]')
        ax.set_title('Current Response I(t)')
    elif plot_type == "Voltage":
        voltage = square_wave(t, T_val, alpha_val)
        line, = ax.plot(t, voltage, lw=2, label="Voltage V(t)")
        ax.set_ylabel('Voltage [V]')
        ax.set_title('Input Square Wave Voltage V(t)')
    else:  # Both
        current = fourier_current(T_val, R_val, L_val, alpha_val, h_val, n_val, t)
        voltage = square_wave(t, T_val, alpha_val)
        ax.plot(t, current, lw=2, label="Current I(t)")
        ax.plot(t, voltage, lw=2, label="Voltage V(t)")
        ax.set_ylabel('Amplitude')
        ax.set_title('Voltage and Current')
    
    ax.set_xlabel('Time [s]')
    ax.grid(True)
    ax.legend()
    
    # Set reasonable y-limits
    if plot_type == "Current" or plot_type == "Both":
        y_max = max(max(fourier_current(T_val, R_val, L_val, alpha_val, h_val, n_val, t)), 10) * 1.1
        ax.set_ylim(0, y_max)
    else:
        ax.set_ylim(-1, 11)
    
    fig.canvas.draw_idle()

# Register the update function with each slider
time_slider.on_changed(update)
alpha_slider.on_changed(update)
R_slider.on_changed(update)
L_slider.on_changed(update)
h_slider.on_changed(update)
n_slider.on_changed(update)
radio.on_clicked(plot_type_update)

# Create a reset button
resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')

def reset(event):
    time_slider.reset()
    alpha_slider.reset()
    R_slider.reset()
    L_slider.reset()
    h_slider.reset()
    n_slider.reset()
    global plot_type
    plot_type = "Current"
    radio.set_active(0)
    update(None)
    
button.on_clicked(reset)

# Add text explaining the equation
equation_text = "Using equations (0.20) and (0.21) from the image"
fig.text(0.5, 0.01, equation_text, ha='center', fontsize=10)

ax.grid(True)
ax.legend()
plt.show()

