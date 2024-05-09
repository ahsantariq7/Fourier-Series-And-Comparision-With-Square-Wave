import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def square_wave_coeff(n):
    if n == 0:
        return 0
    elif n % 2 == 1:
        return 4 / (np.pi * n)
    else:
        return 0


def fourier_series(coeffs, x, L, N):
    series = 0.5 * coeffs[0]
    series_terms = np.zeros((N + 1, len(x)))  # Initialize array to store terms
    series_terms[0] = 0.5 * coeffs[0]
    for n in range(1, N + 1):
        term = coeffs[n] * np.sin(n * np.pi * x / L)
        series_terms[n] = term
        series += term
    return series, series_terms


def square_wave(x):
    return np.where(np.sin(2 * np.pi * x) >= 0, 1, -1)


def plot_fourier_series(x, series_terms, ax, title):
    ax.set_title(title)
    ax.plot(x, square_wave(x), label="Square Wave")
    composite_wave = np.sum(series_terms, axis=0)
    ax.plot(x, composite_wave, label="Fourier Series Approximation")
    ax.legend()


def calculate_rms_error(original_wave, approx_wave):
    differences = original_wave - approx_wave
    squared_differences = differences**2
    mean_squared_difference = np.mean(squared_differences)
    rms_error = np.sqrt(mean_squared_difference)
    return rms_error


L = 1
N = 30
num_points = 1000
x_values = np.linspace(0, L, num_points)
coefficients = [square_wave_coeff(n) for n in range(N + 1)]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.set_facecolor("#f0f0f0")
ax2.set_facecolor("#f0f0f0")

(line,) = ax1.plot([], [], label="Shape", linewidth=2)
ax1.set_xlim(0, L)
ax1.set_ylim(-2, 2)
ax1.grid(True)
ax1.legend()

ax2.set_xlim(0, L)
ax2.set_ylim(-2, 2)
ax2.grid(True)

ax1.set_title("Animating Square Wave using Fourier Series")
ax1.set_xlabel("x")
ax1.set_ylabel("f(x)")

ax2.set_title("Fourier Series vs. Square Wave")
ax2.set_xlabel("x")
ax2.set_ylabel("f(x)")

rms_error_text = ax2.text(
    0.02, 0.95, "", transform=ax2.transAxes, fontsize=10, color="red"
)


def init():
    line.set_data([], [])
    return (line,)


def animate(i):
    line.set_data([], [])
    updated_coefficients = [
        coeff * np.sin((i + 1) * np.pi * (n + 1) / 50)
        for n, coeff in enumerate(coefficients)
    ]
    y_values, series_terms = fourier_series(updated_coefficients, x_values, L, N)
    line.set_data(x_values, y_values)

    # Calculate RMS error
    approx_wave = np.sum(series_terms, axis=0)
    rms_error = calculate_rms_error(square_wave(x_values), approx_wave)
    rms_error_text.set_text(f"RMS Error: {rms_error:.4f}")

    ax2.clear()
    plot_fourier_series(x_values, series_terms, ax2, "Fourier Series vs. Square Wave")

    line.set_color(plt.cm.viridis(i / 200))

    ax1.set_xlim(0, L)
    ax1.set_ylim(-2, 2)

    return (line,)


ani = FuncAnimation(fig, animate, init_func=init, frames=200, interval=200, blit=True)

plt.show()
