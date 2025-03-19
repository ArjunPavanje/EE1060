import matplotlib.pyplot as plt
import numpy as np
import math 
from matplotlib.widgets import Button, Slider
from matplotlib.widgets import RangeSlider

N = 10 # Number of periods to plot
h = 0.01 # Step Size
n = 10000

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

def square_wave(t, T, alpha, amplitude=10):
    """Generate square wave voltage with duty cycle alpha"""
    return amplitude * ((t/T - np.floor(t/T)) < alpha)

def square(t, alpha, T, amplitude=10):
    """Square wave function for numerical methods"""
    return amplitude if (t/T - np.floor(t/T)) < alpha else 0

def backward_euler(T, tau, L, alpha, h, t_values):
    """Backward Euler numerical method implementation"""
    R = L/tau  # Calculate R from tau and L
    fe = []
    y = 0  # Initial current
    x = 0  # Initial time
    
    for i in range(len(t_values)):
        fe.append(y)
        y = ((L*y) + (h*square(x+h, alpha, T)))/(L+h*R)
        x += h
    
    return fe

def forward_euler(T, tau, L, alpha, h, t_values):
    """Forward Euler numerical method implementation"""
    R = L/tau  # Calculate R from tau and L
    fe = []
    y = 0  # Initial current
    x = 0  # Initial time
    
    for i in range(len(t_values)):
        fe.append(y)
        y = y * (1 - (h/tau)) + (h/L) * square(x, alpha, T)
        x += h
    
    return fe

def rk4(T, tau, L, alpha, h, t_values):
    """Runge-Kutta 4th order method implementation"""
    R = L/tau  # Calculate R from tau and L
    result = []
    y = 0  # Initial current
    x = 0  # Initial time
    
    # Define the differential equation: di/dt = (v(t) - R*i)/L = (v(t) - i/tau)/L
    def f(t, i):
        return (square(t, alpha, T) - i/tau)/L
    
    for i in range(len(t_values)):
        result.append(y)
        
        # RK4 algorithm
        k1 = h * f(x, y)
        k2 = h * f(x + h/2, y + k1/2)
        k3 = h * f(x + h/2, y + k2/2)
        k4 = h * f(x + h, y + k3)
        
        y = y + (k1 + 2*k2 + 2*k3 + k4)/6
        x += h
    
    return result

alpha = 0.5 # duty ratio for input square wave
L = 1 # inductance
tau = 1 # time constant L/R
T = 1 # time period of input square wave
plot_type = "All Numerical" # Default plot type

t = np.linspace(0, N*T, int(N*T/h))

# Create the figure and the line that we will manipulate
fig = plt.figure(figsize=(12, 8))
ax = fig.add_axes([0.1, 0.35, 0.8, 0.55])

# Initial plot - all numerical methods
current = fourier_current(T, tau, L, alpha, h, 20, t)
be_current = backward_euler(T, tau, L, alpha, h, t)
fe_current = forward_euler(T, tau, L, alpha, h, t)
rk4_current = rk4(T, tau, L, alpha, h, t)

ax.plot(t, current, lw=2, label="Fourier Series")
ax.plot(t, be_current, lw=2, linestyle='--', label="Backward Euler")
ax.plot(t, fe_current, lw=2, linestyle='-.', label="Forward Euler")
ax.plot(t, rk4_current, lw=2, linestyle=':', label="RK4")

ax.set_xlabel('Time [s]')
ax.set_ylabel('Current [A]')
ax.set_title('RL Circuit Response - Comparison of All Numerical Methods')
ax.grid(True)
ax.legend()

# Make sliders to control parameters
slider_width = 0.65
slider_height = 0.03
slider_left = 0.25
slider_bottom_start = 0.22

axtime = fig.add_axes([slider_left, slider_bottom_start, slider_width, slider_height])
axalpha = fig.add_axes([slider_left, slider_bottom_start - 0.04, slider_width, slider_height])
axtau = fig.add_axes([slider_left, slider_bottom_start - 0.08, slider_width, slider_height])
axL = fig.add_axes([slider_left, slider_bottom_start - 0.12, slider_width, slider_height])
axh = fig.add_axes([slider_left, slider_bottom_start - 0.16, slider_width, slider_height])
axn = fig.add_axes([slider_left, slider_bottom_start - 0.20, slider_width, slider_height])

time_slider = Slider(
    ax=axtime,
    label='Time Period T',
    valmin=0.1,
    valmax=50,
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
    valmax=10,
    valinit=tau,
    valstep=0.01
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
rax = fig.add_axes([0.025, 0.35, 0.15, 0.35])
radio = plt.matplotlib.widgets.RadioButtons(
    rax, ('Fourier', 'Backward Euler', 'Forward Euler', 'RK4', 'All Numerical', 'Voltage', 'All'), active=4)

def plot_type_update(val):
    global plot_type
    plot_type = val
    update(None)

def update(val):
    T_val = time_slider.val
    tau_val = tau_slider.val
    L_val = L_slider.val
    alpha_val = alpha_slider.val
    h_val = h_slider.val
    n_val = n_slider.val
    
    # Update time array if h changed
    global t
    t = np.linspace(0, N*T_val, int(N*T_val/h_val))
    
    # Clear previous plot
    ax.clear()
    
    if plot_type == "Fourier":
        current = fourier_current(T_val, tau_val, L_val, alpha_val, h_val, n_val, t)
        ax.plot(t, current, lw=2, label="Fourier Series")
        ax.set_ylabel('Current [A]')
        ax.set_title('Current Response - Fourier Series')
    elif plot_type == "Backward Euler":
        be_current = backward_euler(T_val, tau_val, L_val, alpha_val, h_val, t)
        ax.plot(t, be_current, lw=2, label="Backward Euler")
        ax.set_ylabel('Current [A]')
        ax.set_title('Current Response - Backward Euler Method')
    elif plot_type == "Forward Euler":
        fe_current = forward_euler(T_val, tau_val, L_val, alpha_val, h_val, t)
        ax.plot(t, fe_current, lw=2, label="Forward Euler")
        ax.set_ylabel('Current [A]')
        ax.set_title('Current Response - Forward Euler Method')
    elif plot_type == "RK4":
        rk4_current = rk4(T_val, tau_val, L_val, alpha_val, h_val, t)
        ax.plot(t, rk4_current, lw=2, label="RK4")
        ax.set_ylabel('Current [A]')
        ax.set_title('Current Response - RK4 Method')
    elif plot_type == "All Numerical":
        current = fourier_current(T_val, tau_val, L_val, alpha_val, h_val, n_val, t)
        be_current = backward_euler(T_val, tau_val, L_val, alpha_val, h_val, t)
        fe_current = forward_euler(T_val, tau_val, L_val, alpha_val, h_val, t)
        rk4_current = rk4(T_val, tau_val, L_val, alpha_val, h_val, t)
        ax.plot(t, current, lw=2, label="Fourier Series")
        ax.plot(t, be_current, lw=2, linestyle='--', label="Backward Euler")
        ax.plot(t, fe_current, lw=2, linestyle='-.', label="Forward Euler")
        ax.plot(t, rk4_current, lw=2, linestyle=':', label="RK4")
        ax.set_ylabel('Current [A]')
        ax.set_title('Current Response - Comparison of All Numerical Methods')
    elif plot_type == "Voltage":
        voltage = square_wave(t, T_val, alpha_val)
        ax.plot(t, voltage, lw=2, label="Voltage V(t)")
        ax.set_ylabel('Voltage [V]')
        ax.set_title('Input Square Wave Voltage V(t)')
    else:  # All
        current = fourier_current(T_val, tau_val, L_val, alpha_val, h_val, n_val, t)
        be_current = backward_euler(T_val, tau_val, L_val, alpha_val, h_val, t)
        fe_current = forward_euler(T_val, tau_val, L_val, alpha_val, h_val, t)
        rk4_current = rk4(T_val, tau_val, L_val, alpha_val, h_val, t)
        voltage = square_wave(t, T_val, alpha_val)
        ax.plot(t, current, lw=2, label="Fourier Series")
        ax.plot(t, be_current, lw=2, linestyle='--', label="Backward Euler")
        ax.plot(t, fe_current, lw=2, linestyle='-.', label="Forward Euler")
        ax.plot(t, rk4_current, lw=2, linestyle=':', label="RK4")
        ax.plot(t, voltage, lw=2, label="Voltage V(t)")
        ax.set_ylabel('Amplitude')
        ax.set_title('Voltage and Current - All Methods')
    
    ax.set_xlabel('Time [s]')
    ax.grid(True)
    ax.legend()
    
#    # Set fixed y-limits
#    if plot_type == "Voltage":
#        #ax.set_ylim(-1, 11)
#    else:
#        #ax.set_ylim(0, 20)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw_idle()

# Register the update function with each slider
time_slider.on_changed(update)
alpha_slider.on_changed(update)
tau_slider.on_changed(update)
L_slider.on_changed(update)
h_slider.on_changed(update)
n_slider.on_changed(update)
radio.on_clicked(plot_type_update)

# Create a reset button
resetax = fig.add_axes([0.8, 0.25, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')

def reset(event):
    time_slider.reset()
    alpha_slider.reset()
    tau_slider.reset()
    L_slider.reset()
    h_slider.reset()
    n_slider.reset()
    global plot_type
    plot_type = "All Numerical"
    radio.set_active(4)  # Set to "All Numerical" by default
    update(None)
    
button.on_clicked(reset)

# Add text explaining the equation
equation_text = "Response of series RL circuit to square wave input"
fig.text(0.5, 0.01, equation_text, ha='center', fontsize=10)

plt.show()

