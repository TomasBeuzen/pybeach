# -*- coding: utf-8 -*-
"""
Created on Wed Mar 6 12:43:40 2019

@author: Tomas Beuzen

Functions for handling classifiers used to support pybeach.
"""
import joblib
import pkg_resources
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from pybeach.support import data_support as ds


def create_classifier(x, z, toe, window=40, min_buffer=40, max_buffer=200):
    """
    Create dune toe classifier.

    ...

    Parameters
    ----------
    x : ndarray
        Array of cross-shore locations of size (m,).
    z : ndarray
        Array of elevations matching x. May be of size (m,) or (n,m).
    toe : ndarray
        Array of dune toe locations of size (n,).
    window : int, default 40
        Size of the window for training data.
    min_buffer : int, default 40
        Minimum buffer around the real dune toe.
    max_buffer : int, default 200
        Maximum buffer range.

    Returns
    -------
    clf : scikit-learn classifier
        Created random forest classifier.
    """

    # Pre-processing
    z = ds.interp_nan(x, z)  # interp nan
    xx = np.arange(np.min(x), np.max(x) + 0.5, 0.5)
    z = ds.interp_to_grid(x, xx, z)  # interp to grid
    toe = ds.interp_toe_to_grid(x, xx, toe)
    z = ds.moving_average(z, 5)  # apply moving average to smooth
    z = ds.diff_data(z, 1)  # differentiate

    # Create data
    features, labels = create_training_data(xx, z, toe, window, min_buffer, max_buffer)

    # Build classifier
    clf = RandomForestClassifier(
        n_estimators=100, criterion="gini", random_state=123
    ).fit(features, labels.ravel())
    return clf


def load_classifier(clf_name):
    """
    Load classifier.

    ...

    Parameters
    ----------
    clf_name : str
        Name of classifier to load.

    Returns
    -------
    clf : scikit-learn classifier
        Classifier.
    """
    clf_path = pkg_resources.resource_filename(
        "pybeach", "classifiers/" + clf_name + ".joblib"
    )
    with open(clf_path, "rb") as f:
        clf = joblib.load(f)
    return clf


def create_training_data(x, z, toe, window=40, min_buffer=40, max_buffer=200):
    """
    Create training data to develop dune toe classifier.

    ...

    Parameters
    ----------
    x : ndarray
        Array of cross-shore locations of size (m,).
    z : ndarray
        Array of elevations matching x. May be of size (m,) or (m,n).
    toe : ndarray
        Array of dune toe locations of size (n,).
    window : int, default 40
        Size of the window for training data.
    min_buffer : int, default 40
        Minimum buffer around the real dune toe.
    max_buffer : int, default 200
        Maximum buffer range.

    Returns
    -------
    features : ndarray
        Samples of size window centered around a true or false dune toe.
    labels : bool
        Label of 0 (not a toe) or 1 (toe)
    """
    # Initialize
    z_pos = np.zeros((len(z), window * 2 + 1))
    lab_pos = np.ones((len(z), 1))
    z_neg = np.zeros((len(z), window * 2 + 1))
    lab_neg = np.zeros((len(z), 1))
    indices = np.arange(0, len(x))
    # Loop
    for i in np.arange(0, len(z)):
        # Store positive example
        z_pos[i] = z[i, toe[i] - window : toe[i] + window + 1]
        # Find points away from pos example
        rand_ind = np.where(
            (abs(indices - toe[i]) > min_buffer)
            & (abs(indices - toe[i]) < max_buffer)
            & (indices > window)
            & (indices < len(x) - window)
        )[0]
        # Select a random one
        rand_ind = np.random.choice(rand_ind, 1)[0]
        # Store negative example
        z_neg[i] = z[i, rand_ind - window : rand_ind + window + 1]
    # Merge data
    features = np.concatenate([z_pos, z_neg], axis=0)
    labels = np.concatenate([lab_pos, lab_neg], axis=0)
    return features, labels
