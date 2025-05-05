import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Define the step function (f(t))
def step_function(t):
    return np.heaviside(t, 1)  # Step function u(t)

# Define the rectangular kernel function (h(t))
def rectangle_kernel(t, T):
    return np.where(np.abs(t) <= T, 1, 0)  # Rectangular kernel for -T <= t <= T

# Convolution function (f * h)(t)
def convolution(f, h, t):
    # Ensure the convolution is normalized by the time step
    return np.convolve(f, h, mode='same') * (t[1] - t[0])

# Time range
t_min = -10
t_max = 10
t = np.linspace(t_min, t_max, 1000)

# Initial values for T and T0
T_init = 2
T0_init = 0

# Create the step function signal
f_t = step_function(t)

# Create the initial rectangular kernel
h_t = rectangle_kernel(t - T0_init, T_init)

# Plot the initial convolution
conv_t = convolution(f_t, h_t, t)

#  Correct the initial convolution as well
conv_t[t >= (T0_init + T_init)] = 2 * T_init

# Set up the figure and axes
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.25)
ax.plot(t, f_t, label="Step function f(t)", lw=2)
ax.plot(t, h_t, label="Rectangular Kernel h(t)", lw=2)
ax.plot(t, conv_t, label="Convolution y(t)", lw=2, linestyle="--")
ax.set_title("Convolution of Step Function with Rectangular Kernel")
ax.set_xlabel("Time (t)")
ax.set_ylabel("Amplitude")
ax.legend(loc="upper left")

# Add sliders for T and T0
ax_T = plt.axes([0.1, 0.01, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_T0 = plt.axes([0.1, 0.06, 0.65, 0.03], facecolor='lightgoldenrodyellow')

slider_T = Slider(ax_T, 'T', 0.1, 5.0, valinit=T_init)
slider_T0 = Slider(ax_T0, 'T0', -5.0, 5.0, valinit=T0_init)

# Update function for the sliders
def update(val):
    T = slider_T.val
    T0 = slider_T0.val
    h_t = rectangle_kernel(t - T0, T)
    conv_t = convolution(f_t, h_t, t)

    # After t > (T0 + T), the convolution should be constant at 2T
    conv_t[t >= (T0 + T)] = 2 * T

    ax.clear()
    ax.plot(t, f_t, label="Step function f(t)", lw=2)
    ax.plot(t, h_t, label="Rectangular Kernel h(t)", lw=2)
    ax.plot(t, conv_t, label="Convolution y(t)", lw=2, linestyle="--")
    ax.set_title("Convolution of Step Function with Rectangular Kernel")
    ax.set_xlabel("Time (t)")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper left")
    fig.canvas.draw_idle()

# Connect the sliders to the update function
slider_T.on_changed(update)
slider_T0.on_changed(update)

plt.show()
