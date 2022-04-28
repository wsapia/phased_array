# Created By    : Weston Sapia
# Created Date  : Apr-12-2022

"""Simulate both an electrical and mechanical scan of an antenna and plot the results to compare"""

import numpy as np
import pandas as pd
import pylab as plt

from common import electrical_scan, mechanical_scan

# Array constants
N = 8
FREQ = 10.18e9
D = 0.014
STEERING_ANGLE = 0

# Set the scan angles for the measurements
SCAN_EXTENT = 90
SCAN_ANGLES = np.arange(-SCAN_EXTENT, SCAN_EXTENT + 0.01, 0.1)

# Function for the element factor
EF = True
EF_FUNCTION = np.cos if EF is True else lambda x: np.ones_like(x)

# Function to taper the array
TAPER = True
TAPER_FUNCTION = np.blackman if TAPER is True else np.ones

# Boolean to show the plot during data collection (Runs a bit slower, speed it up by limiting the number of shown steps)
SHOW_PLOT = True
MAX_NUM_STEPS = 161

# Boolean to display the data in polar form
POLAR = False


def compare_scans(angles: np.ndarray, steering_angle: float = 0):
    """Calculate the array pattern for an electrical scan

    Args:
        angles (np.ndarray): NumPy array of angles to scan over (in degrees).
        steering_angle (float): Steering angle of the AUT.
        linear (bool): Boolean indicating whether to return linear or log data
    """

    # Convert the angles to radians
    rads = np.deg2rad(angles)

    # Set the x axis depending on the plot type
    x = angles
    if POLAR:
        x = rads

    # Create a fixed, mechanically scanned array for measuring
    fixed_arr = mechanical_scan(
        angles=angles,
        steering_angle=steering_angle,
        linear=True,
        num_elements=N,
        d=D,
        frequency=FREQ,
        ef=EF_FUNCTION,
        taper=TAPER_FUNCTION,
    )

    if SHOW_PLOT:

        # Create a handle for the electrical scan generator
        elec = electrical_scan(
            angles=angles,
            steering_angle=steering_angle,
            linear=True,
            num_elements=N,
            d=D,
            frequency=FREQ,
            ef=EF_FUNCTION,
            taper=TAPER_FUNCTION,
            show_steps=True,
            num_steps=MAX_NUM_STEPS,
        )

        fig = plt.figure(
            "Mechanical vs. Electrical Scan (Linear, Normalized)", figsize=(14, 5.5)
        )

        # Get the axis handle
        if POLAR:
            ax = fig.add_subplot(projection="polar")
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
        else:
            ax = fig.add_subplot()

        # Plot the AUT
        ax.plot(x, np.abs(fixed_arr), label="Mechanical")

        # Plot dummy data for the probe and the integral and retrieve the plot handles
        (p1,) = ax.plot(x, np.ones_like(angles), label="Electrical")
        (p2,) = ax.plot(x, np.ones_like(angles), label="Probe")
        p3 = ax.axvline(-400, c="k", ls="--", alpha=0.4)
        ax.plot(
            x,
            TAPER_FUNCTION(rads.size) * EF_FUNCTION(rads),
            "-.",
            label="Taper & Element Factor",
        )

        # Set the limits and titles of the plot
        ax.set_xlim(x[0], x[-1])
        if POLAR:
            ax.set_xticks(np.deg2rad(np.arange(-SCAN_EXTENT, SCAN_EXTENT + 1, 10)))
        else:
            ax.set_xticks(np.arange(-SCAN_EXTENT, SCAN_EXTENT + 1, 10))
        ax.set_ylim(0, 1.05)
        ax.set_title("Probe Steering Angle: -1.00°")

        # Turn on the grid and legend
        ax.grid(1)
        ax.legend()

        # Tighten up the layout
        fig.tight_layout()

        # Set the plot to interactive mode (for updating on the fly) & show the plot window
        plt.ion()
        plt.show()

        # Go through each yielded point in the scan
        for angle, meas_arr, data in elec:

            # Normalize the data
            norm = data / np.max(np.abs(np.nan_to_num(data)))

            # Show the scan's progress
            p1.set_data(x, np.abs(norm))
            p2.set_data(x, np.abs(meas_arr))
            p3.set_xdata(np.deg2rad(angle))

            # Update the title
            ax.set_title(f"Probe steering angle: {angle:>6.2f}°")

            # Redraw the plot
            fig.canvas.draw()
            fig.canvas.flush_events()

    else:
        data = next(
            electrical_scan(
                angles=angles,
                steering_angle=steering_angle,
                linear=True,
                num_elements=N,
                d=D,
                frequency=FREQ,
                ef=EF_FUNCTION,
                taper=TAPER_FUNCTION,
                show_steps=False,
            )
        )

    # Normalize both sets of data
    fixed_arr /= np.max(np.abs(fixed_arr))
    data /= np.max(np.abs(data))

    # Convert to dB
    fixed_db = 10 * np.log10(fixed_arr**2)
    data_db = 10 * np.log10(data**2)

    """
    # The element factor & taper pull down the peak when the steering angle is off of boresight.
    # Normalize the mechanical scan, as this is how we would plot the data in the real world
    # if we had measured it in a chamber
    """
    fixed_db -= np.max(fixed_db)

    if SHOW_PLOT:
        # Disable the interactive display
        plt.ioff()

        # Close the current figure
        plt.close(fig)

    # Create the figure for plotting
    if POLAR:
        fig, (ax1, ax2) = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(14, 5.5),
            tight_layout=1,
            subplot_kw={"projection": "polar"},
        )
    else:
        fig, (ax1, ax2) = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(14, 5.5),
            tight_layout=1,
        )

    # Title the figure
    fig.canvas.manager.set_window_title("Mechanical vs. Electrical Scan")

    # Plot the linear data
    (m,) = ax1.plot(x, np.abs(fixed_arr), label="Mechanical")
    (e,) = ax1.plot(x, np.abs(data), label="Electrical")
    ax1.set_title("Normalized Linear")
    ax1.set_ylim(0, 1.1)

    # Plot the log data
    ax2.plot(x, fixed_db, label="Mechanical")
    ax2.plot(x, data_db, label="Electrical")
    ax2.set_title("Normalized Log")
    if POLAR:
        ax2.set_ylim(-30, 2)
    else:
        ax2.set_ylim(-60, 2)

    # Show the measured data if the angles match up
    if STEERING_ANGLE in (0, 31.5):

        # Pull in the measured data
        meas_data = pd.read_csv(f"meas_{STEERING_ANGLE:.1f}_deg.csv")
        m_angles = meas_data["m_angle"].values
        m_mags = meas_data["m_mag"].values
        e_angles = meas_data["e_angle"].values
        e_mags = meas_data["e_mag"].values

        # Remove any NaNs
        m_angles = m_angles[np.isfinite(m_angles)]
        e_angles = e_angles[np.isfinite(e_angles)]
        m_mags = m_mags[np.isfinite(m_mags)]
        e_mags = e_mags[np.isfinite(e_mags)]

        # Normalize the data
        m_mags -= np.max(m_mags)
        e_mags -= np.max(e_mags)

        # Calculate the linear values
        m_lin = 10 ** (m_mags / 20)
        e_lin = 10 ** (e_mags / 20)

        # Convert to radians
        if POLAR:
            m_angles = np.deg2rad(m_angles)
            e_angles = np.deg2rad(e_angles)

        # Add the traces
        ax1.plot(
            m_angles, m_lin, "--", color=m.get_color(), label="Mechanical Measured"
        )
        ax1.plot(
            e_angles, e_lin, "--", color=e.get_color(), label="Electrical Measured"
        )
        ax2.plot(
            m_angles, m_mags, "--", color=m.get_color(), label="Mechanical Measured"
        )
        ax2.plot(
            e_angles, e_mags, "--", color=e.get_color(), label="Electrical Measured"
        )

    # Tidy up the axes
    for ax in (ax1, ax2):
        ax.set_xlim(x[0], x[-1])
        if POLAR:
            ax.set_xticks(np.deg2rad(np.arange(-SCAN_EXTENT, SCAN_EXTENT + 1, 10)))
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
        else:
            ax.set_xticks(np.arange(-SCAN_EXTENT, SCAN_EXTENT + 1, 10))
        ax.grid(1)
        ax.legend()

    # Show the plot
    plt.show()


if __name__ == "__main__":

    # Compare the mechanical and electrical scans
    compare_scans(SCAN_ANGLES, STEERING_ANGLE)
