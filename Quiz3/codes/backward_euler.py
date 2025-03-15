import matplotlib.pyplot as plt
import numpy as np
import math 
from matplotlib.widgets import Button, Slider
from matplotlib.widgets import RangeSlider

N = 10 # Number of points of output wave to plot
h = 0.01 # Step Size

def square(t, alpha, T, A=10):
    if 0 < t%T < alpha*T:
        return A 
    else: 
        return 0


def backward_euler(T, R, L, alpha, h):
    fe = []
    t1 = []
    y = x = 0
    for i in t:
        fe.append(y)
        t1.append(x)
        #np.append(fe, y)
        #np.append(t1, x)
        #t1[i] = x
        #fe[i] = y
        y = ((L*y) + (h*square(x+h, alpha, T)) )/(L+h*R)
        x+=h 
    return fe

alpha = 0.5 # duty ratio for input square wave
R = 1 # resistance
L = 1 # inductance
T = 1 # time period of input square wave

t = np.linspace(0, N*T, int(N*T/h))
# Create the figure and the line that we will manipulate
fig, ax = plt.subplots()
line, = ax.plot(t, backward_euler(T, R, L, alpha, h), lw=2, label = "Backward Euler")
ax.set_xlabel('Time [s]')

# adjust the main plot to make room for the sliders
fig.subplots_adjust(left=0.25, bottom=0.35)


# Make a horizontal slider to control the frequency.
axtime = fig.add_axes([0.25, 0.1, 0.65, 0.03])
axalpha = fig.add_axes([0.25, 0.15, 0.65, 0.03])
axR = fig.add_axes([0.25, 0.20, 0.65, 0.03])
axL = fig.add_axes([0.25, 0.25, 0.65, 0.03])
axh = fig.add_axes([0.25, 0.30, 0.65, 0.03])
time_slider = Slider(
    ax=axtime,
    label='Time Period',
    valmin=0.1,
    valmax=30,
    valinit=T,
)
alpha_slider = Slider(
    ax=axalpha,
    label='Alpha',
    valmin=0,
    valmax=1,
    valinit=alpha,
)
R_slider = Slider(
    ax=axR,
    label='Resistance R',
    valmin=1e-4,
    valmax=1e4,
    valinit=R,
)
L_slider = Slider(
    ax=axL,
    label='inductance L',
    valmin=1e-4,
    valmax=1,
    valinit=L,
)
h_slider = Slider(
    ax=axh,
    label='Step size h',
    valmin=1e-5,
    valmax=1e-2,
    valinit=h,
)
def update(val):
    line.set_ydata(backward_euler(time_slider.val, R_slider.val, L_slider.val, alpha_slider.val, h_slider.val))
    fig.canvas.draw_idle()
    #line.set_label("Backward Euler")  # Add this line to reset the label
    #ax.legend()  # Update the legend

# register the update function with each slider
time_slider.on_changed(update)
alpha_slider.on_changed(update)
R_slider.on_changed(update)
L_slider.on_changed(update)
h_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')


def reset(event):
    time_slider.reset()
    alpha_slider.reset()
    R_slider.reset()
    L_slider.reset()
    h_slider.reset()

button.on_clicked(reset)
ax.set_ylim(0, 20)
ax.legend()
plt.show()

