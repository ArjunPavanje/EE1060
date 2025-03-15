import matplotlib.pyplot as plt
import numpy as np
import math 
from matplotlib.widgets import Button, Slider

N = 1000  # Number of points of output wave to plot
h = 0.01  # Step Size

def square(t, alpha, T, A=10):
    if 0 < t % T < alpha * T:
        return A
    else: 
        return 0

def forward_euler(T, R, L, alpha):
    fe = np.zeros(N)
    t1 = np.zeros(N)
    y = x = 0
    for i in range(0, N):
        t1[i] = x
        fe[i] = y
        y += h * (square(x, alpha, T) - (y * R)) / L
        x += h
    return fe

alpha = 0.5  # duty ratio for input square wave
R = 1  # resistance
L = 1  # inductance
T = 1  # time period of input square wave

# t = np.linspace(0, N * T, int(N * T / h))

t = np.linspace(0, N * h, N)  # Create time array with N points (matches forward_euler's output)

# Create the figure and the line that we will manipulate
fig, ax = plt.subplots()
line, = ax.plot(t, forward_euler(T, R, L, alpha), lw=2)
ax.set_xlabel('Time [s]')

# Adjust the main plot to make room for the sliders
fig.subplots_adjust(left=0.25, bottom=0.25)

# Make a horizontal slider to control the time period.
axperiod = fig.add_axes([0.25, 0.1, 0.65, 0.03])
time_slider = Slider(
    ax=axperiod,
    label='Time Period',
    valmin=0.1,
    valmax=30,
    valinit=T,
)

def update(val):
    # Recompute the y-values for the plot with the new time period
    line.set_ydata(forward_euler(time_slider.val, R, L, alpha))
    ax.set_xlim(0, N * time_slider.val)  # Adjust x-axis according to the new time period
    fig.canvas.draw_idle()

# Register the update function with the slider
time_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')

def reset(event):
    time_slider.reset()

button.on_clicked(reset)

plt.show()

