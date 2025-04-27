
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from matplotlib.widgets import RadioButtons

# 1. Setup time vector
t = np.linspace(-4*np.pi, 4*np.pi, 10000)

# 2. Define signals
def generate_signal(signal_type):
    if signal_type == 'Sine':
        return np.sin(t)
    elif signal_type == 'Exponential':
        return np.exp(0.3 * t) / np.exp(0.3 * 2*np.pi) * 2 - 1  # normalized to [-1,1]
    else:
        raise ValueError(f"Unknown signal type: {signal_type}")

# 3. Define noise
def add_noise(signal):
    noise = 0.3 * np.random.randn(len(t))
    return signal + noise

# 4. Define box kernel
T = 30  # width parameter (in samples)
kernel_size = 2 * T + 1
box_kernel = np.ones(kernel_size) / kernel_size  # normalize it

# 5. Create the plot
fig, ax = plt.subplots(figsize=(12, 6))
plt.subplots_adjust(left=0.3)  # leave space on left for buttons
noisy_line, = ax.plot([], [], label='Noisy Signal', alpha=0.5)
smoothed_line, = ax.plot([], [], label='Denoised Signal', color='red')
clean_line, = ax.plot([], [], label='True Signal', color='green', linestyle='dashed')

ax.legend()
ax.set_title('Denoising Noisy Signal Using Box Kernel')
ax.set_xlabel('Time')
ax.set_ylabel('Amplitude')
ax.grid(True)

# 6. Radio buttons
ax_radio = plt.axes([0.05, 0.4, 0.15, 0.15])  # [left, bottom, width, height]
radio = RadioButtons(ax_radio, ('Sine', 'Exponential'))

# 7. Update function
def update(signal_type):
    clean_signal = generate_signal(signal_type)
    noisy_signal = add_noise(clean_signal)
    smoothed_signal = convolve(noisy_signal, box_kernel, mode='same')
    
    clean_line.set_data(t, clean_signal)
    noisy_line.set_data(t, noisy_signal)
    smoothed_line.set_data(t, smoothed_signal)
    
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw_idle()

# 8. Connect the button
radio.on_clicked(update)

# 9. Initialize
update('Sine')

plt.show()
