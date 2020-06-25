# -*- coding: utf-8 -*-
"""
Created on Wed Mar 6 12:43:40 2019

@author: Tomas Beuzen

Functions for processing data used to support pybeach.
"""

import numpy as np
import pandas as pd
import warnings


def interp_nan(x, z):
    """
    Linear interpolation over nan values. End points are set to nearest non-nan value.

    ...

    Parameters
    ----------
    x : ndarray
        Array of cross-shore locations of size (m,).
    z : ndarray
        Array of elevations matching x. May be of size (m,) or (m,n).

    Returns
    -------
    z_interp : ndarray
        Interpolated z.

    """
    z_interp = np.array(
        [np.interp(x, x[~np.isnan(row)], row[~np.isnan(row)]) for row in z]
    )
    return z_interp


def interp_to_grid(x, x_interp, z):
    """
    Interpolates coordinates to a 0.5m grid.

    ...

    Parameters
    ----------
    x : ndarray
        Array of cross-shore locations of size (m,).
    x_interp : ndarray
        Array of cross-shore coordinates to interpolate to.
    z : ndarray
        Array of elevations matching x. May be of size (m,) or (m,n).

    Returns
    -------
    x_interp : ndarray
        x interpolated to 0.5m cross-shore spacing.
    z_interp : ndarray
        z interpolated to 0.5m cross-shore spacing.

    """
    z_interp = np.array([np.interp(x_interp, x, row) for row in z])
    return z_interp


def interp_toe_to_grid(x, x_interp, toe):
    """
    Interpolates dune toe location from x coordinates to x_interp coordinates.

    ...

    Parameters
    ----------
    x : ndarray
        Array of cross-shore coordinates.
    x_interp : ndarray
        Array of cross-shore coordinates to interpolate toe locations to.
    toe : ndarray
        Array of dune toe locations on x coordinates, to be interpolated to x_interp coordinates.

    Returns
    -------
    toe_interp : ndarray
        toe interpolated to x_interp.

    """
    toe_interp = np.array([np.abs(x_interp - x[_]).argmin() for _ in toe])

    return toe_interp


def moving_average(z, window_size=5):
    """
    Smoothes z using a moving average.

    ...

    Parameters
    ----------
    z : ndarray
        Array of elevations. May be of size (m,) or (m,n).
    window_size : int, default 5
        Size of smoothing window.

    Returns
    -------
    z_smooth : ndarray
        z smoothed.

    """
    assert (
        isinstance(window_size, int) & (window_size > 0) & (window_size < z.shape[1])
    ), f"window_size must be int between 0 and {z.shape[1]}."

    z_smooth = np.array(
        [
            np.convolve(
                np.pad(
                    row,
                    (window_size // 2, window_size - 1 - window_size // 2),
                    mode="edge",
                ),
                np.ones((window_size,)) / window_size,
                mode="valid",
            )
            for row in z
        ]
    )
    return z_smooth


def diff_data(z, diff_order=1):
    """
    Differentiate z.

    ...

    Parameters
    ----------
    z : ndarray
        Array of elevations. May be of size (m,) or (m,n).
    diff_order : int, default 1
        Degree of differentiation.

    Returns
    -------
    z_diff : ndarray
        z differentiated.

    """
    z_diff = np.array([np.pad(np.diff(row, diff_order), (0, 1), "edge") for row in z])
    return z_diff


def rolling_samples(z, window_size):
    """
    Create samples of size window_size from z. End points re padded with zeros so that
    samples can be made at the end points.

    ...

    Parameters
    ----------
    z : ndarray
        Array of elevations. May be of size (m,) or (m,n).
    window_size : int
        Size of sample window.

    Returns
    -------
    z_samples : ndarray
        Samples from z.

    """
    z = np.pad(
        z,
        (window_size // 2, window_size - 1 - window_size // 2),
        mode="constant",
        constant_values=0,
    )  # pad to catch end points
    shape = z.shape[:-1] + (z.shape[-1] - window_size + 1, window_size)
    strides = z.strides + (z.strides[-1],)
    z_samples = np.lib.stride_tricks.as_strided(z, shape=shape, strides=strides)
    return z_samples


def relative_relief(z, window_size, water_level):
    """
    Calculate relative relief at each cross-shore coordinate of a profile. Based on
    Wernette et al. (2016):

    Wernette, P., Houser, C., & Bishop, M. P. (2016). An automated approach for
    extracting Barrier Island morphology from digital elevation models.
    Geomorphology, 262, 1-7.

    ...

    Parameters
    ----------
    z : ndarray
        Array of elevations. May be of size (m,) or (m,n).
    window_size : int
        Size of sample window.
    water_level : number
        Elevation of mean water level, profile elevations at or below mean water
        level elevation are set to NaN to ensure stability of relative relief
        calculations.

    Returns
    -------
    z_rr : ndarray
        Relative relief of z.

    """
    window_size = np.atleast_1d(window_size)
    rr = np.zeros((window_size.shape[0], z.shape[0]))
    zz = pd.Series(z).copy()
    zz[zz <= water_level] = np.nan  # remove topography below water level
    for i, w in enumerate(window_size):
        z_max = zz.rolling(w, center=True).max()
        z_min = zz.rolling(w, center=True).min()
        rr[i, :] = (zz - z_min) / (z_max - z_min)
    # suppress RuntimeWarnings in np.nanmean when requesting mean over only nan values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z_rr = np.nanmean(rr, axis=0)
    return z_rr
