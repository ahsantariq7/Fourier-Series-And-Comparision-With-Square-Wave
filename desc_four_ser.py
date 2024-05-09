import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Define the Fourier series coefficients for a square wave
def square_wave_coeff(n):
    if n == 0:
        return 0
    elif n % 2 == 1:
        return 4 / (np.pi * n)
    else:
        return 0


# Calculate Fourier series
def fourier_series(coeffs, x, L, N):
    series = 0.5 * coeffs[0]
    series_terms = [0.5 * coeffs[0]]
    for n in range(1, N + 1):
        term = coeffs[n] * np.sin(n * np.pi * x / L)
        series_terms.append(term)
        series += term
    return series, series_terms


# Calculate Root Mean Square (RMS) error
def calculate_rms_error(original, approx):
    return np.sqrt(np.mean((original - approx) ** 2))


# Parameters
L = 1  # Length of the square wave
N = 50  # Number of terms in the Fourier series
num_points = 1000  # Number of points to plot
x_values = np.linspace(0, L, num_points)
coefficients = [square_wave_coeff(n) for n in range(N + 1)]

# Create figure and axis objects
fig, ax = plt.subplots()
ax.set_facecolor("#f0f0f0")  # Set light background color
(line,) = ax.plot([], [], label="Shape", linewidth=2)  # Increase line thickness
ax.set_xlim(0, L)
ax.set_ylim(-2, 2)
ax.grid(True)

# Add labels and values
ax.set_title("Animating Square Wave using Fourier Series")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
text = ax.text(0.02, 0.95, "", transform=ax.transAxes)
coeff_text = ax.text(0.02, 0.90, "", transform=ax.transAxes)
rms_error_text = ax.text(0.02, 0.85, "", transform=ax.transAxes, color="red")

# Add legend
ax.legend()


# Initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line, text, coeff_text, rms_error_text


def animate(i):
    # Clear the previous plot
    line.set_data([], [])

    # Generate y values using Fourier series with updated coefficients
    updated_coefficients = [
        coeff * np.sin((i + 1) * np.pi * (n + 1) / 50)
        for n, coeff in enumerate(coefficients)
    ]
    y_values, series_terms = fourier_series(updated_coefficients, x_values, L, N)
    line.set_data(x_values, y_values)
    text.set_text(f"Frame: {i}")
    coeff_text.set_text(f"Fourier Series: {series_terms}")

    # Calculate RMS error
    original_wave = np.where(np.sin(2 * np.pi * x_values) >= 0, 1, -1)
    rms_error = calculate_rms_error(original_wave, y_values)
    rms_error_text.set_text(f"RMS Error: {rms_error:.4f}")

    # Find the highest peak value and its index
    peak_index = np.argmax(y_values)
    highest_peak_value = y_values[peak_index]

    # Remove previous annotations
    for annotation in ax.texts:
        annotation.remove()

    # Add annotations for some specific points
    if i % 1 == 0:  # Adjust the interval for annotations
        for j in range(
            0, len(x_values), len(x_values) // 30
        ):  # Adjust the number of annotations
            ax.annotate(
                f"{y_values[j]:.2f}",
                (x_values[j], y_values[j]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )

    # Annotate highest peak value
    ax.annotate(
        f"{highest_peak_value:.2f}",
        (x_values[peak_index], highest_peak_value),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontsize=10,
        color="red",
    )

    # Set a different color for each frame
    line.set_color(plt.cm.viridis(i / 200))  # Adjust colormap as needed

    # Adjust axis limits for more clearance
    ax.set_xlim(0, L)
    ax.set_ylim(-2, 2)

    return line, text, coeff_text, rms_error_text


# Create animation
ani = FuncAnimation(fig, animate, init_func=init, frames=200, interval=200, blit=True)

plt.show()
