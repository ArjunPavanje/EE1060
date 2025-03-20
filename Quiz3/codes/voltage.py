
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
    
def inductor_voltage(current,L,h):
    v_L=np.zeros_like(current)
    for i in range(1,len(current)):
        v_L[i]=L*(current[i]-current[i-1])/h
    return v_L
    
alpha = 0.5 # duty ratio for input square wave
L = 1 # inductance
tau = 1 # time constant L/R
T = 1 # time period of input square wave
plot_type = "Fourier" # Default plot type

t = np.linspace(0, N*T, int(N*T/h))

# Create the figure and the line that we will manipulate
fig = plt.figure(figsize=(12, 8))
ax = fig.add_axes([0.1, 0.35, 0.8, 0.55])

# Initial plot - all numerical methods
fe_current = fourier_current(T, tau, L, alpha, h, 20, t)
v_L = inductor_voltage(fe_current, L, h)

ax.plot(t, v_L, lw=2, label="Forward Euler - Inductor Voltage")
ax.set_xlabel('Time [s]')
ax.set_ylabel('Voltages')
ax.set_title('RL Circuit Response - Fourier')
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
    rax, ('Fourier', 'Backward Euler', 'Forward Euler', 'RK4', 'All Numerical', 'Input', 'All'), active=0)

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
    

    global t
    t = np.linspace(0, N*T_val, int(N*T_val/h_val))  # Update time array
    
    ax.clear()  # Clear previous plot

    v_L = np.array([0])  # Ensure v_L is always initialized

    if plot_type == "Fourier":
        current = fourier_current(T_val, tau_val, L_val, alpha_val, h_val, n_val, t)
        v_L = inductor_voltage(current, L_val, h_val)
        ax.plot(t, v_L, lw=2, label="Fourier Series - Inductor Voltage")

    elif plot_type == "Backward Euler":
        be_current = backward_euler(T_val, tau_val, L_val, alpha_val, h_val, t)
        v_L = inductor_voltage(be_current, L_val, h_val)
        ax.plot(t, v_L, lw=2, label="Backward Euler - Inductor Voltage")

    elif plot_type == "Forward Euler":
        fe_current = forward_euler(T_val, tau_val, L_val, alpha_val, h_val, t)
        v_L = inductor_voltage(fe_current, L_val, h_val)
        ax.plot(t, v_L, lw=2, label="Forward Euler - Inductor Voltage")

    elif plot_type == "RK4":
        rk4_current = rk4(T_val, tau_val, L_val, alpha_val, h_val, t)
        v_L = inductor_voltage(rk4_current, L_val, h_val)
        ax.plot(t, v_L, lw=2, label="RK4 - Inductor Voltage")

    elif plot_type == "Input":
        voltage = square_wave(t, T_val, alpha_val)
        ax.plot(t, voltage, lw=2, label="Input voltage V(t)")
        ax.set_ylabel('Input Voltage [V]')
        ax.set_title('Input Square Wave Voltage V(t)')
        v_L = voltage  # Assign a placeholder value to v_L
        
    elif plot_type == "All Numerical":
        current = fourier_current(T_val, tau_val, L_val, alpha_val, h_val, n_val, t)
        v_L = inductor_voltage(current, L_val, h_val)
        be_current = backward_euler(T_val, tau_val, L_val, alpha_val, h_val, t)
        be_v_L = inductor_voltage(be_current, L_val, h_val)
        fe_current = forward_euler(T_val, tau_val, L_val, alpha_val, h_val, t)
        fe_v_L = inductor_voltage(fe_current, L_val, h_val)
        rk4_current = rk4(T_val, tau_val, L_val, alpha_val, h_val, t)
        rk4_v_L = inductor_voltage(rk4_current, L_val, h_val)
     
        ax.plot(t, v_L, lw=2, label="Fourier Series - Inductor Voltage")
        ax.plot(t, be_v_L, lw=2, label="Backward Euler - Inductor Voltage")
        ax.plot(t, fe_v_L, lw=2, label="Forward Euler - Inductor Voltage")
        ax.plot(t, rk4_v_L, lw=2, label="RK4 - Inductor Voltage")

    else:  # All (Voltage + Current)
        current = fourier_current(T_val, tau_val, L_val, alpha_val, h_val, n_val, t)
        v_L = inductor_voltage(current, L_val, h_val)
        be_current = backward_euler(T_val, tau_val, L_val, alpha_val, h_val, t)
        be_v_L = inductor_voltage(be_current, L_val, h_val)
        fe_current = forward_euler(T_val, tau_val, L_val, alpha_val, h_val, t)
        fe_v_L = inductor_voltage(fe_current, L_val, h_val)
        rk4_current = rk4(T_val, tau_val, L_val, alpha_val, h_val, t)
        rk4_v_L = inductor_voltage(rk4_current, L_val, h_val)
        voltage = square_wave(t, T_val, alpha_val)
        
        ax.plot(t, voltage, lw=2, label="Input Voltage V(t)")
        ax.plot(t, v_L, lw=2, label="Fourier Series - Inductor Voltage")
        ax.plot(t, be_v_L, lw=2, label="Backward Euler - Inductor Voltage")
        ax.plot(t, fe_v_L, lw=2, label="Forward Euler - Inductor Voltage")
        ax.plot(t, rk4_v_L, lw=2, label="RK4 - Inductor Voltage")

    ax.set_ylim(min(v_L) - abs(min(v_L)), max(v_L) + abs(max(v_L)))

    ax.set_xlabel('Time [s]')
    ax.grid(True)
    ax.legend()

    # Set fixed y-limits
    if plot_type == "Voltage":
        ax.set_ylim(-1, 11)
    else:
        ax.set_ylim(-15, 15)

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
    plot_type = "Fourier"
    update(None)
    radio.set_active(0)
button.on_clicked(reset)

# Add text explaining the equation
equation_text = "Using equations (0.20) and (0.21) from the image with τ = L/R"
fig.text(0.5, 0.01, equation_text, ha='center', fontsize=10)

update(None)
plt.show()

