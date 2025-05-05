
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from sympy import symbols, integrate, Poly

# Define the polynomial function
def polynomial_function(t, coeffs, T0):
    shifted_t = t - T0
    return sum(c * shifted_t**i for i, c in enumerate(reversed(coeffs)))

# Analytical convolution polynomial coefficients
def analytical_convolution_poly(coeffs, T, T0, kernel_type='full'):
    t, u = symbols('t u')
    n = len(coeffs) - 1
    f_u = sum(c * (u - T0)**i for i, c in enumerate(reversed(coeffs)))
    
    if kernel_type == 'full':
        # Full box kernel from t-T to t+T
        y_t = integrate(f_u, (u, t - T, t + T))
    elif kernel_type == 'shifted':
        # Shifted box kernel from t to t+2T
        y_t = integrate(f_u, (u, t, t + 2*T))
    elif kernel_type == 'positive_half':
        # Positive half box kernel from t to t+T
        y_t = integrate(f_u, (u, t, t + T))
    else:
        raise ValueError("Invalid kernel_type. Choose 'full', 'shifted', or 'positive_half'.")
    
    p = Poly(y_t, t)
    conv_coeffs = p.all_coeffs()
    conv_coeffs = [float(c) for c in conv_coeffs]
    return conv_coeffs

# Evaluate polynomial at points t
def evaluate_poly(t, coeffs, T0):
    shifted_t = t - T0
    return sum(c * shifted_t**i for i, c in enumerate(reversed(coeffs)))

# Initial parameters
n_init = 2
T_init = 2
T0_init = 0

t_min = -15
t_max = 15
t = np.linspace(t_min, t_max, 1000)

# Initial coefficients
coeffs_init = [1.0] + [0.0] * n_init

# Set up figure and axis
fig, ax = plt.subplots(figsize=(12, 8))
plt.subplots_adjust(left=0.25, right=0.75, bottom=0.25)

# Initial plot data
f_t = polynomial_function(t, coeffs_init, T0_init)
conv_coeffs_full = analytical_convolution_poly(coeffs_init, T_init, T0_init, 'full')
conv_coeffs_shifted = analytical_convolution_poly(coeffs_init, T_init, T0_init, 'shifted')
conv_coeffs_positive_half = analytical_convolution_poly(coeffs_init, T_init, T0_init, 'positive_half')

conv_t_full = evaluate_poly(t, conv_coeffs_full, T0_init)
conv_t_shifted = evaluate_poly(t, conv_coeffs_shifted, T0_init)
conv_t_positive_half = evaluate_poly(t, conv_coeffs_positive_half, T0_init)

# Plot lines
line_f, = ax.plot(t, f_t, label='Original Polynomial', lw=2, color='#20b2aa')
line_conv_full, = ax.plot(t, conv_t_full, label='Convolution', lw=2, linestyle='--', color='#ffa500')
line_conv_shifted, = ax.plot(t, conv_t_shifted, label='Shifted Box Kernel', lw=2, linestyle=':', color='#ff6347')
line_conv_positive_half, = ax.plot(t, conv_t_positive_half, label='t>0', lw=2, linestyle='-.', color='#9370db')

ax.set_xlabel('t')
ax.set_ylabel('Amplitude')
ax.legend()
ax.set_title('Analytical Convolution of Polynomial with Rectangular Kernel')
ax.grid(True, alpha=0.3)

# Create sliders for T, T0, and degree n
ax_T = plt.axes([0.25, 0.15, 0.5, 0.03], facecolor='lightgoldenrodyellow')
ax_T0 = plt.axes([0.25, 0.10, 0.5, 0.03], facecolor='lightgoldenrodyellow')
ax_n = plt.axes([0.25, 0.05, 0.5, 0.03], facecolor='lightgoldenrodyellow')

slider_T = Slider(ax_T, 'T', 0.1, 5.0, valinit=T_init)
slider_T0 = Slider(ax_T0, 'T0', -5.0, 5.0, valinit=T0_init)
slider_n = Slider(ax_n, 'Degree n', 0, 10, valinit=n_init, valstep=1)

# Lists to hold coefficient sliders and their axes
slider_coeffs = []
slider_axes = []

# Function to create coefficient sliders based on degree
def create_coeff_sliders(n):
    global slider_coeffs, slider_axes
    # Remove old sliders
    for ax_slider in slider_axes:
        ax_slider.remove()
    slider_coeffs = []
    slider_axes = []

    start_y = 0.75
    slider_height = 0.03
    spacing = 0.04

    for i in range(n + 1):
        y_pos = start_y - i * spacing
        ax_slider = plt.axes([0.8, y_pos, 0.15, slider_height], facecolor='lightgoldenrodyellow')
        # Set initial value to 1 for highest degree term, 0 for others
        init_val = 1.0 if i == 0 else 0.0
        slider = Slider(ax_slider, f'a{n-i}', -5.0, 5.0, valinit=init_val)
        slider_axes.append(ax_slider)
        slider_coeffs.append(slider)
        slider.on_changed(update)

# Update function for sliders
def update(val):
    n = int(slider_n.val)
    T = slider_T.val
    T0 = slider_T0.val

    coeffs = [s.val for s in slider_coeffs]

    f_t = polynomial_function(t, coeffs, T0)
    conv_coeffs_full = analytical_convolution_poly(coeffs, T, T0, 'full')
    conv_coeffs_shifted = analytical_convolution_poly(coeffs, T, T0, 'shifted')
    conv_coeffs_positive_half = analytical_convolution_poly(coeffs, T, T0, 'positive_half')
    
    conv_t_full = evaluate_poly(t, conv_coeffs_full, T0)
    conv_t_shifted = evaluate_poly(t, conv_coeffs_shifted, T0)
    conv_t_positive_half = evaluate_poly(t, conv_coeffs_positive_half, T0)

    line_f.set_ydata(f_t)
    line_conv_full.set_ydata(conv_t_full)
    line_conv_shifted.set_ydata(conv_t_shifted)
    line_conv_positive_half.set_ydata(conv_t_positive_half)

    ax.set_xlim(T0 - 15, T0 + 15)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw_idle()

# Update degree slider callback
def update_degree(val):
    n = int(slider_n.val)
    create_coeff_sliders(n)
    update(val)

# Connect sliders
slider_T.on_changed(update)
slider_T0.on_changed(update)
slider_n.on_changed(update_degree)

# Create initial coefficient sliders
create_coeff_sliders(n_init)

plt.show()
