
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

# Define the Fourier spectrum function
def fourier_spectrum(n, alpha):
    Cn = (20/(np.pi *n))*(np.sin(np.pi*n*alpha))
    return Cn

# Define the Fourier phase spectrum function
def fourier_phase(n, alpha):
    phase = np.arctan(np.tan(n*np.pi*alpha))
    return phase

# Plotting the spectra
def plot_spectrum(alpha):
    n = np.arange(1, 50, 1)
    amplitude = [fourier_spectrum(i, alpha) for i in n]
    phase_values = [fourier_phase(i, alpha) for i in n]

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(left=0.1, bottom=0.4)

    # Initial plot (default to magnitude)
    stem_container = ax.stem(n, amplitude)
    ax.set_xlabel('n')
    ax.set_ylabel('Fourier Coefficient')
    ax.set_title('Amplitude Spectrum')
    ax.grid(True)

    # Slider axes
    ax_alpha = plt.axes([0.1, 0.2, 0.65, 0.03], facecolor='lightgoldenrodyellow')

    # Reset button
    reset_button_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
    reset_button = Button(reset_button_ax, 'Reset', color='lightgoldenrodyellow', hovercolor='0.975')

    # Plot type radio buttons
    plot_type_ax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor='lightgoldenrodyellow')
    plot_type_radio = RadioButtons(plot_type_ax, ('Magnitude', 'Phase'))

    # Create slider
    alpha_slider = Slider(ax_alpha, 'Alpha', 0.01, 0.99, valinit=alpha, valstep=0.01)

    def update(val=None):
        alpha = alpha_slider.val

        # Recompute values
        amplitude = [fourier_spectrum(i, alpha) for i in n]
        phase_values = [fourier_phase(i, alpha) for i in n]

        # Clear the plot
        ax.cla()

        # Plot based on selected plot type
        plot_type = plot_type_radio.value_selected
        
        if plot_type == 'Magnitude':
            ax.stem(n, amplitude)
            ax.set_ylabel('Fourier Coefficient')
            ax.set_title('Amplitude Spectrum')
        else:  # Phase
            ax.stem(n, phase_values)
            ax.set_ylabel('Phase (radians)')
            ax.set_title('Phase Spectrum')
        
        ax.set_xlabel('n')
        ax.grid(True)
        fig.canvas.draw_idle()

    def reset(event):
        alpha_slider.reset()
        update()

    # Connect callbacks
    alpha_slider.on_changed(update)
    reset_button.on_clicked(reset)
    plot_type_radio.on_clicked(update)

    plt.show()

# Initial call
plot_spectrum(0.5)
