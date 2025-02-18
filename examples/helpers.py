"""Helper functions for working with the HeatDiffusion model."""

import matplotlib.pyplot as plt
import numpy


def plot_temperature(temperature: numpy.ndarray, plate_size: int) -> None:
    """Plot the plate temperature with a colorbar."""
    fig, ax = plt.subplots()
    tp = ax.imshow(temperature)
    ax.set_title("Plate Temperature")
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_xlim(
        [temperature.shape[0] // 2 - plate_size, temperature.shape[0] // 2 + plate_size]
    )
    ax.set_ylim(
        [temperature.shape[1] // 2 - plate_size, temperature.shape[1] // 2 + plate_size]
    )
    fig.colorbar(tp, label="Temperature (C)")
