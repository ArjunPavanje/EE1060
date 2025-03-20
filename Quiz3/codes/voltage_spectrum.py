import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

'''Phase and Magnitude Spectrum of fourier coeffecients of input square wave'''

# Defining the Fourier spectrum function
def fourier_spectrum(n, alpha, omega_0, tau, L):
    # R is calculated from tau and L
    R = L / tau
    Cn = abs(20/(np.pi*n)*np.sin(n*np.pi*alpha))
    return Cn

# Defining the Fourier phase spectrum function
def fourier_phase(n, alpha):
    phase = -np.arctan(np.tan(np.pi*n*alpha))
    return phase

# Plotting the spectra
def plot_spectrum(alpha, omega_0, tau, L):
    n = np.arange(1, 50, 1)
    amplitude = [fourier_spectrum(i, alpha, omega_0, tau, L) for i in n]
    phase_values = [fourier_phase(i, alpha) for i in n]

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(left=0.15, bottom=0.4, right=0.85)

    # Initial plot (default to magnitude)
    stem_container = ax.stem(n, amplitude)
    ax.set_xlabel('n')
    ax.set_ylabel('Fourier Coefficient')
    ax.set_title('Amplitude Spectrum')
    ax.grid(True)

    # Slider axes
    ax_alpha = plt.axes([0.15, 0.3, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    ax_omega_0 = plt.axes([0.15, 0.25, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    ax_tau = plt.axes([0.15, 0.2, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    ax_L = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor='lightgoldenrodyellow')

    # Reset button - moved to bottom left
    reset_button_ax = plt.axes([0.15, 0.05, 0.1, 0.04])
    reset_button = Button(reset_button_ax, 'Reset', color='lightgoldenrodyellow', hovercolor='0.975')

    # Plot type radio buttons - moved to bottom right
    plot_type_ax = plt.axes([0.85, 0.15, 0.1, 0.15], facecolor='lightgoldenrodyellow')
    plot_type_radio = RadioButtons(plot_type_ax, ('Magnitude', 'Phase'))

    # Create sliders
    alpha_slider = Slider(ax_alpha, 'Alpha', 0.01, 0.99, valinit=alpha, valstep=0.01)
    omega_0_slider = Slider(ax_omega_0, 'ω₀', 0.1, 10, valinit=omega_0, valstep=0.1)
    tau_slider = Slider(ax_tau, 'τ', 0.1, 10, valinit=tau, valstep=0.01)
    L_slider = Slider(ax_L, 'L', 0.01, 10, valinit=L, valstep=0.01)

    def update(val=None):
        alpha = alpha_slider.val
        omega_0 = omega_0_slider.val
        tau = tau_slider.val
        L = L_slider.val

        # Recompute values
        amplitude = [fourier_spectrum(i, alpha, omega_0, tau, L) for i in n]
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
        omega_0_slider.reset()
        tau_slider.reset()
        L_slider.reset()
        update()

    # Connect callbacks
    alpha_slider.on_changed(update)
    omega_0_slider.on_changed(update)
    tau_slider.on_changed(update)
    L_slider.on_changed(update)
    reset_button.on_clicked(reset)
    plot_type_radio.on_clicked(update)

    plt.show()

# Initial call with default parameters
plot_spectrum(0.5, 1, 1, 1)
