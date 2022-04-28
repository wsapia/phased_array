# Created By    : Weston Sapia
# Created Date  : Apr-12-2022

"""Common functions for use in other programs"""

from inspect import currentframe
from typing import Callable, Iterable, Union
from warnings import filterwarnings

import numpy as np
from scipy.constants import speed_of_light


def find_closest(available, actual: Union[float, Iterable]):
    if isinstance(actual, Iterable):
        actual = np.array(actual)
        return np.array(
            available[np.argmin(np.abs(actual[:, None] - available), axis=1)]
        )
    else:
        return available[np.argmin(np.abs(available - actual))]


def electrical_scan(
        angles: np.ndarray,
        steering_angle: float = 0,
        linear: bool = False,
        num_elements: int = 8,
        d: float = 0.014,
        frequency: float = 10.18e9,
        ef: Callable = np.cos,
        taper: Callable = np.hamming,
        show_steps: bool = False,
        num_steps: int = 101
):
    """Calculate the array pattern for an electrical scan

    Args:
        angles (np.ndarray): NumPy array of angles to scan over (in degrees).
        steering_angle (float): Steering angle of the AUT.
        linear (bool): Boolean indicating whether to return the data in linear format. If False, return log data.
        num_elements (int): Number of elements in the array
        d (float): Element spacing
        frequency (float): Frequency of operation
        ef (Callable): Element factor function
        taper (Callable): Taper function
        show_steps (bool): Boolean indicating whether to yield the data as it's calculated
        num_steps (int): Number of steps to show during calculation
    """

    # Create a fixed array for measuring
    fixed_arr = mechanical_scan(
        angles=angles,
        steering_angle=steering_angle,
        linear=True,
        num_elements=num_elements,
        d=d,
        frequency=frequency,
        ef=ef,
        taper=taper
    )

    # Angles to update the plots
    if not 0 < num_steps <= angles.size:
        num_steps = angles.size
    plot_update_angles = list(angles[::angles.size // num_steps]) + [angles[-1]]

    # Create an array for storing the result of each measurement
    data = np.full(angles.size, np.nan)

    for i, angle in enumerate(angles):
        # Measurement array steered to the angle of interest
        meas_arr = mechanical_scan(
            angles,
            angle,
            linear=True,
            num_elements=num_elements,
            d=d,
            frequency=frequency,
            ef=ef,
            taper=taper
        )

        # Integral of the data (This is how an RF power detector would operate)
        data[i] = np.trapz(fixed_arr * meas_arr, angles)

        # Yield the data as it sits
        if show_steps and angle in plot_update_angles:
            yield angle, meas_arr, data

    if not show_steps:
        if linear:
            yield data
        else:
            yield 10 * np.log10(np.power(data, 2))


def mechanical_scan(
        angles: np.ndarray,
        steering_angle: float = 0,
        linear: bool = False,
        num_elements: int = 8,
        d: float = 0.014,
        frequency: float = 10.18e9,
        ef: Callable = np.cos,
        taper: Callable = np.hamming
):
    """Calculate the array pattern for a mechanical scan

    Args:
        angles (np.ndarray): NumPy array of angles to scan over (in degrees).
        steering_angle (float): Steering angle of the AUT.
        linear (bool): Boolean indicating whether to return the data in linear format. If False, return log data.
        num_elements (int): Number of elements in the array
        d (float): Element spacing
        frequency (float): Frequency of operation
        ef (function): Element factor function
        taper (function): Taper function
    """

    wl = speed_of_light / frequency

    # Calculate the delta phi for the elements due to the steering angle
    delta_phi = 2 * np.pi * d * np.sin(np.deg2rad(steering_angle)) / wl

    # Ignore the warning about invalid values when the angle matches the intended peak (divide by 0)
    filterwarnings("ignore", lineno=currentframe().f_lineno + 4)

    # Calculate the array factor across the angles of interest
    common = (np.pi * d / wl) * np.sin(np.deg2rad(angles)) - (delta_phi / 2)
    array_factor = np.sin(num_elements * common) / (num_elements * np.sin(common))

    # Convert any NaNs to 1 (happens at the peak when the ignored warning above happens)
    array_factor = np.nan_to_num(array_factor, nan=1)

    # Taper the array
    array_factor *= taper(array_factor.size)

    # Create the element factor (assuming patch antennas)
    element_factor = ef(np.deg2rad(angles))

    # Combine the antenna factor and element factor
    array = array_factor * element_factor

    if linear:
        return array
    else:
        return 10 * np.log10(np.power(array, 2))
