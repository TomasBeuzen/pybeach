# -*- coding: utf-8 -*-
"""
Created on Wed Mar 6 12:43:40 2019

@author: Tomas Beuzen
"""

import numpy as np
from pybeach.support import data_support as ds
from pybeach.support import classifier_support as cs


class Profile:
    def __init__(self, x, z, window_size=5):
        """
        A class used to represent a 2D beach profile transect.

        ...

        Parameters
        ----------
        x : ndarray
            Array of cross-shore locations of size (m,).
        z : ndarray
            Array of elevations matching x. May be of size (m,) or (m,n).
        window_size : int, default 5
            Size of window used to smooth z with a moving average.

        Attributes
        ----------
        x_orig : ndarray
            Original input array of cross-shore locations.
        z_orig : ndarray
            Original array of profile elevations matching x_orig.
        x : ndarray
            x_orig interpolated to 0.5 m grid.
        z : ndarray
            z_orig interpolated to 0.5 m grid and smoothed by a moving average with
            window size smooth_window.

        Methods
        -------
        predict_dunetoe_ml(self, clf_name, no_of_output=1, dune_crest='rr', **kwargs)
        predict_dunetoe_mc(self, dune_crest='rr', shoreline=True, window_size=None, **kwargs)
        predict_dunetoe_pd(self, dune_crest=None, shoreline=None, **kwargs)
        predict_dunetoe_rr(self, window_size=11, threshold=0.2, water_level=0)
        predict_dunecrest(self, method="max", window_size=50, threshold=0.8, water_level=0)
        predict_shoreline(self, water_level=0, dune_crest='rr', **kwargs)

        """
        assert isinstance(x, np.ndarray) & (
            np.ndim(x) == 1
        ), "x should be of type ndarray and shape (m,)."
        assert np.ndim(x) == 1, "x should be a 1-d array of size (m,)."
        assert len(x) > 1, "x should have length > 1."
        assert isinstance(z, np.ndarray), "z should be of type ndarray."
        assert (
            isinstance(window_size, int) & (window_size > 0) & (window_size < len(x))
        ), f"window_size must be int between 0 and {len(x)}."

        # Ensure inputs are row vectors
        x = np.atleast_1d(x)
        z = np.atleast_2d(z)
        if len(x) not in z.shape:
            raise ValueError(
                f"Input x of shape ({x.shape[0]},) must share a dimension with input z which has shape {z.shape[0], z.shape[1]}."
            )
        if x.shape[0] != z.shape[1]:
            z = z.T

        # Store original inputs
        self.x = x
        self.z = z

        # Interp nan values
        z = ds.interp_nan(x, z)
        flag = np.polyfit(x, z.T, 1)[0]
        if np.any(flag > 0):
            raise Warning(
                f"Input profiles should be oriented from landward (left) to seaward (right), "
                f"some inputted profiles appear to have the sea on the left. This may cause errors."
            )

        # Interp to 0.5 m grid
        self.x_interp = np.arange(np.min(x), np.max(x) + 0.5, 0.5)
        z = ds.interp_to_grid(x, self.x_interp, z)

        # Apply moving average to smooth data
        z = ds.moving_average(z, window_size)

        # Store transformed inputs
        self.z_interp = z

    def predict_dunetoe_ml(self, clf_name, no_of_output=1, dune_crest="max", **kwargs):
        """
        Predict dune toe location using a pre-trained machine learning (ml) classifier.
        See pybeach/classifiers/create_classifier.py to create a classifier.

        ...

        Parameters
        ----------
        clf_name : str
            Classifier to use. Classifier should be contained within 'classifiers'
            directory. In-built options include "barrier", "embayed", "mixed".
        no_of_output : int, default 1
            Number of dune toes to return, ranked from most probable to least probable.
        dune_crest : {'max', 'rr', int, None}, default 'max'
            Method to identify the dune crest location. The region of the beach profile
            that the dune toe location is searched for is constrained to the region
            seaward of the dune crest.
            max: the maximum elevation of the cross-shore profile.
            rr: dune crest calculated based on relative relief.
            int: integer specifying the location of the dune crest. Of size 1 or
                 self.z.shape[0].
            None: do not calculate a dune crest location. Search the whole profile for
                  the dune toe.
        **kwargs : arguments
            Additional arguments to pass to `self.predict_dunecrest()`. Keywords include
            window_size, threshold, water_level.

        Returns
        -------
        dt_index : array of ints
            array containing the indices of no_of_outputs dune toe locations, in
            descending order of probability.
        dt_probabilities : array
            array of dune toe probabilities for each profiles in self.z.

        """
        # Warnings
        assert isinstance(clf_name, str), "clf_name should be a string."
        assert (
            isinstance(no_of_output, int)
            & (no_of_output > 0)
            & (no_of_output < len(self.x_interp))
        ), f"no_of_outputs must be int between 0 and {len(self.x)}."

        # Define dune crest
        if dune_crest in ["max", "rr"]:
            for k in kwargs.keys():
                if k not in ["window_size", "threshold", "water_level"]:
                    raise Warning(f"{k} not a valid argument for predict_dunecrest()")
            kwargs = {
                k: v
                for k, v in kwargs.items()
                if k in ["window_size", "threshold", "water_level"]
            }
            dune_crest_loc = self.predict_dunecrest(method=dune_crest, **kwargs)
        elif isinstance(dune_crest, int):
            dune_crest_loc = np.full((self.z_interp.shape[0],), dune_crest)
        elif dune_crest is None:
            dune_crest_loc = np.full((self.z_interp.shape[0],), 0)
        elif len(dune_crest) == self.z_interp.shape[0] & isinstance(
            dune_crest, np.ndarray
        ) & all(isinstance(_, np.int64) for _ in dune_crest):
            dune_crest_loc = dune_crest
        else:
            raise ValueError(
                f'dune_crest should be "max", "rr", int (of size 1 or {self.z_interp.shape[0]}), or None'
            )

        # Load the random forest classifier
        try:
            clf = cs.load_classifier(clf_name)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"no classifier named {clf_name} found in classifier folder."
            )

        # Differentiate data
        z_diff = ds.diff_data(self.z_interp, 1)

        # Predict probability of dune toe for all points along profile
        dt_probabilities = np.array(
            [
                clf.predict_proba(np.squeeze(ds.rolling_samples(row, clf.n_features_)))[
                    :, 1
                ]
                for row in z_diff
            ]
        )

        # Interpolate the probabilities back to the original grid
        dt_probabilities = ds.interp_to_grid(self.x_interp, self.x, dt_probabilities)

        # Retrieve the top 'no_of_outputs' predictions in order
        dt_index = np.array(
            [
                np.flip(np.argsort(row[crest:])[-no_of_output:], 0)
                for row, crest in zip(dt_probabilities, dune_crest_loc)
            ]
        )
        dt_index = np.squeeze(dt_index) + dune_crest_loc

        return dt_index, dt_probabilities

    def predict_dunetoe_mc(
        self, dune_crest="max", shoreline=True, hanning_window=None, **kwargs
    ):
        """
        Predict dune toe location based on profile curvature (mc). Based on
        Stockdon et al. (2007):

        Stockdon, H. F., Sallenger Jr, A. H., Holman, R. A., & Howd, P. A. (2007). A
        simple model for the spatially-variable coastal response to hurricanes. Marine
        Geology, 238(1-4), 1-20.

        ...

        Parameters
        ----------
        dune_crest : {'max', 'rr', int, None}, default 'max'
            Method to identify the dune crest location. The region of the beach profile
            that the dune toe location is searched for is constrained to the region
            seaward of the dune crest.
            max: the maximum elevation of the cross-shore profile.
            rr: dune crest calculated based on relative relief.
            int: integer specifying the location of the dune crest. Of size 1 or
                 self.z.shape[0].
            None: do not calculate a dune crest location. Search the whole profile for
                  the dune toe.
        shoreline : int or bool, default True
            Location of shoreline. The region of the beach profile that the dune toe
            location is searched for is constrained to the region landward of the
            shoreline.
            True: use `predict_shoreline()` to calculate shoreline location.
            False: do not use or find a shoreline location.
            int: integer specifying the location of the shoreline. Of size 1 or
                 self.z.shape[0].
        hanning_window : int, default None
            Size of Hanning window for additional smoothing of profile transect.
        **kwargs : arguments
            Additional arguments to pass to `predict_dunecrest()` and/or
            `predict_shoreline()`. Keywords include window_size, threshold, water_level.

        Returns
        -------
        dt_index : array of ints
            dune toe location.

        """
        if hanning_window:
            assert (
                isinstance(hanning_window, int)
                & (hanning_window > 0)
                & (hanning_window < self.z_interp.shape[1])
            ), f"window_size must be int between 0 and {self.z_interp.shape[1]}."
            window = np.divide(
                np.hanning(hanning_window), np.sum(np.hanning(hanning_window))
            )
        else:
            window = 1

        if dune_crest in ["max", "rr"]:
            for k in kwargs.keys():
                if k not in ["window_size", "threshold", "water_level"]:
                    raise Warning(
                        f"{k} not a valid argument for predict_dunecrest() or predict_shoreline()"
                    )
            kwargs = {
                k: v
                for k, v in kwargs.items()
                if k in ["window_size", "threshold", "water_level"]
            }
            dune_crest_loc = self.predict_dunecrest(method=dune_crest, **kwargs)
        elif isinstance(dune_crest, int):
            dune_crest_loc = np.full((self.z_interp.shape[0],), dune_crest)
        elif dune_crest is None:
            dune_crest_loc = np.full((self.z_interp.shape[0],), 0)
        elif len(dune_crest) == self.z_interp.shape[0] & isinstance(
            dune_crest, np.ndarray
        ) & all(isinstance(_, np.int64) for _ in dune_crest):
            dune_crest_loc = dune_crest
        else:
            raise ValueError(
                f'dune_crest should be "max", "rr", int (of size 1 or {self.z_interp.shape[0]}), or None'
            )

        if shoreline == True:
            for k in kwargs.keys():
                if k not in ["window_size", "threshold", "water_level"]:
                    raise Warning(
                        f"{k} not a valid argument for predict_dunecrest() or predict_shoreline()"
                    )
            kwargs = {k: v for k, v in kwargs.items() if k in ["water_level"]}
            shoreline_loc = self.predict_shoreline(**kwargs)
        elif shoreline == False:
            shoreline_loc = np.full((self.z_interp.shape[0],), -1)
        elif isinstance(shoreline, int):
            shoreline_loc = np.full((self.z_interp.shape[0],), shoreline)
        elif len(shoreline) == self.z_interp.shape[0] & isinstance(
            shoreline, np.ndarray
        ) & all(isinstance(_, np.int64) for _ in shoreline):
            shoreline_loc = shoreline
        else:
            raise ValueError(
                f"shoreline should be bool, or int (of size 1 or {self.z_interp.shape[0]})"
            )

        # Check for shoreline landward of dune crest
        mask = (dune_crest_loc - shoreline_loc) >= 0
        shoreline_loc[mask] = -1

        # Filter with convolution, then find max curvature
        dt_index = np.array(
            [
                np.argmax(
                    np.gradient(
                        np.gradient(
                            np.convolve(row[ind_dunecrest:ind_mwl], window, "same"), 0.5
                        ),
                        0.5,
                    )
                )
                + ind_dunecrest
                for (row, ind_dunecrest, ind_mwl) in zip(
                    self.z, dune_crest_loc, shoreline_loc
                )
            ]
        )

        return dt_index

    def predict_dunetoe_pd(self, dune_crest="max", shoreline=True, **kwargs):
        """
        Predict location of dune toe based as the location of maximum perpendicular
        distance (pd) from the line drawn between the dune crest and shoreline.

        ...

        Parameters
        ----------
        dune_crest : {'max', 'rr', int, None}, default 'max'
            Method to identify the dune crest location. The region of the beach profile
            that the dune toe location is searched for is constrained to the region
            seaward of the dune crest.
            max: the maximum elevation of the cross-shore profile.
            rr: dune crest calculated based on relative relief.
            int: integer specifying the location of the dune crest. Of size 1 or
                 self.z.shape[0].
            None: do not calculate a dune crest location. Search the whole profile for
                  the dune toe.
        shoreline : int or bool, default True
            Location of shoreline. The region of the beach profile that the dune toe
            location is searched for is constrained to the region landward of the
            shoreline.
            True: use `predict_shoreline()` to calculate shoreline location.
            False: do not use or find a shoreline location.
            int: integer specifying the location of the shoreline. Of size 1 or
                 self.z.shape[0].
        **kwargs : arguments
            Additional arguments to pass to `predict_dunecrest()` and/or
            `predict_shoreline()`. Keywords include window_size, threshold, water_level.

        Returns
        -------
        dt_index : array of ints
            dune toe location.

        """
        if dune_crest in ["max", "rr"]:
            for k in kwargs.keys():
                if k not in ["window_size", "threshold", "water_level"]:
                    raise Warning(
                        f"{k} not a valid argument for predict_dunecrest() or predict_shoreline()"
                    )
            kwargs = {
                k: v
                for k, v in kwargs.items()
                if k in ["window_size", "threshold", "water_level"]
            }
            dune_crest_loc = self.predict_dunecrest(method=dune_crest, **kwargs)
        elif isinstance(dune_crest, int):
            dune_crest_loc = np.full((self.z_interp.shape[0],), dune_crest).astype(int)
        elif dune_crest is None:
            dune_crest_loc = np.full((self.z_interp.shape[0],), 0)
        elif len(dune_crest) == self.z_interp.shape[0] & isinstance(
            dune_crest, np.ndarray
        ) & all(isinstance(_, np.int64) for _ in dune_crest):
            dune_crest_loc = dune_crest.astype(int)
        else:
            raise ValueError(
                f'dune_crest should be "max", "rr", int (of size 1 or {self.z_interp.shape[0]}), or None'
            )

        if shoreline == True:
            for k in kwargs.keys():
                if k not in ["window_size", "threshold", "water_level"]:
                    raise Warning(
                        f"{k} not a valid argument for predict_dunecrest() or predict_shoreline()"
                    )
            kwargs = {k: v for k, v in kwargs.items() if k in ["water_level"]}
            shoreline_loc = self.predict_shoreline(**kwargs)
        elif shoreline is False or shoreline is None:
            shoreline_loc = np.full((self.z_interp.shape[0],), -1).astype(int)
        elif isinstance(shoreline, int):
            shoreline_loc = np.full((self.z_interp.shape[0],), shoreline).astype(int)
        elif len(shoreline) == self.z_interp.shape[0] & isinstance(
            shoreline, np.ndarray
        ) & all(isinstance(_, np.int64) for _ in shoreline):
            shoreline_loc = shoreline
        else:
            raise ValueError(
                f"shoreline should be bool, or int (of size 1 or {self.z_interp.shape[0]})"
            )

        dt_index = np.full(self.z.shape[0], np.nan)
        for j, row in enumerate(self.z):
            p1 = np.array([self.x[dune_crest_loc[j]], row[dune_crest_loc[j]]])
            p2 = np.array([self.x[shoreline_loc[j]], row[shoreline_loc[j]]])
            p3 = np.array(
                [
                    self.x[dune_crest_loc[j] : shoreline_loc[j]],
                    row[dune_crest_loc[j] : shoreline_loc[j]],
                ]
            )
            dist = np.cross(p2 - p1, p3.T - p1) / np.linalg.norm(p2 - p1)
            # only interested in negative distances (below the line joining the crest and shoreline
            dt_index[j] = np.argmin(dist) + dune_crest_loc[j]

        return dt_index.astype(int)

    def predict_dunetoe_rr(
        self,
        toe_window_size=21,
        toe_threshold=0.2,
        water_level=0,
        dune_crest=None,
        shoreline=None,
        verbose=True,
        **kwargs,
    ):
        """
        Predict dune toe location based on relative relief (rr). Based on
        Wernette et al. (2016):

        Wernette, P., Houser, C., & Bishop, M. P. (2016). An automated approach for
        extracting Barrier Island morphology from digital elevation models.
        Geomorphology, 262, 1-7.

        ...

        Parameters
        ----------
        toe_window_size : int, default 21
            Size of window for calculating relative relief. May be int or list of ints.
            If a list is passed, relative relief is calculated for each window size and
            averaged. See Wernette et al. (2016) for further details.
        toe_threshold : float, default 0.2
            Threshold of relative relief that identifies the dune toe. Between 0 and 1.
            See Wernette et al. (2016) for further details.
        water_level : number, default 0
            Elevation of mean water level, profile elevations at or below mean water
            level elevation are set to NaN to ensure stability of relative relief
            calculations.  See Wernette et al. (2016) for further details.
        dune_crest : {'max', 'rr', int, None}, default 'max'
            Method to identify the dune crest location. The region of the beach profile
            that the dune toe location is searched for is constrained to the region
            seaward of the dune crest.
            max: the maximum elevation of the cross-shore profile.
            rr: dune crest calculated based on relative relief.
            int: integer specifying the location of the dune crest. Of size 1 or
                 self.z.shape[0].
            None: do not calculate a dune crest location. Search the whole profile for
                  the dune toe.
        shoreline : int or bool, default True
            Location of shoreline. The region of the beach profile that the dune toe
            location is searched for is constrained to the region landward of the
            shoreline.
            True: use `predict_shoreline()` to calculate shoreline location.
            False: do not use or find a shoreline location.
            int: integer specifying the location of the shoreline. Of size 1 or
                 self.z.shape[0].
        verbose : bool, default True
            If True, will output notifications for when no relative relief value lies
            below toe_threshold, in which case the minimum relative relief is selected
            as the toe location.
        **kwargs : arguments
            Additional arguments to pass to `predict_dunecrest()` and/or
            `predict_shoreline()`. Keywords include window_size, threshold, water_level.

        Returns
        -------
        dt_index : array of ints
            dune toe location.

        """
        if isinstance(toe_window_size, int):
            assert (
                isinstance(toe_window_size, int)
                & (toe_window_size > 0)
                & (toe_window_size < self.z_interp.shape[1])
            ), f"window_size must be int between 0 and {self.z_interp.shape[1]}."
        elif isinstance(toe_window_size, list):
            assert all(
                isinstance(_, int) & (_ > 0) & (_ < self.z_interp.shape[1])
                for _ in toe_window_size
            ), f"window_size must be int between 0 and {self.z_interp.shape[1]}."
        else:
            raise ValueError(f"window_size must bt of type int or list.")
        assert isinstance(toe_threshold, (int, float)) & (
            toe_threshold > 0 and toe_threshold < 1
        ), f"threshold should be number between 0 and 1, but {toe_threshold} was passed."
        assert isinstance(
            water_level, (int, float)
        ), f"water_level should be a number, but {water_level} was passed."

        if dune_crest in ["max", "rr"]:
            for k in kwargs.keys():
                if k not in ["window_size", "threshold", "water_level"]:
                    raise Warning(
                        f"{k} not a valid argument for predict_dunecrest() or predict_shoreline()"
                    )
            kwargs = {
                k: v
                for k, v in kwargs.items()
                if k in ["window_size", "threshold", "water_level"]
            }
            dune_crest_loc = self.predict_dunecrest(method=dune_crest, **kwargs)
            dune_crest_loc = ds.interp_toe_to_grid(
                self.x, self.x_interp, dune_crest_loc
            )
        elif isinstance(dune_crest, int):
            dune_crest_loc = np.full((self.z_interp.shape[0],), dune_crest).astype(int)
        elif dune_crest is None:
            dune_crest_loc = np.full((self.z_interp.shape[0],), 0)
        elif len(dune_crest) == self.z_interp.shape[0] & isinstance(
            dune_crest, np.ndarray
        ) & all(isinstance(_, np.int64) for _ in dune_crest):
            dune_crest_loc = dune_crest.astype(int)
            dune_crest_loc = ds.interp_toe_to_grid(
                self.x, self.x_interp, dune_crest_loc
            )
        else:
            raise ValueError(
                f'dune_crest should be "max", "rr", int (of size 1 or {self.z_interp.shape[0]}), or None'
            )

        if shoreline == True:
            for k in kwargs.keys():
                if k not in ["window_size", "threshold", "water_level"]:
                    raise Warning(
                        f"{k} not a valid argument for predict_dunecrest() or predict_shoreline()"
                    )
            kwargs = {
                k: v
                for k, v in kwargs.items()
                if k in ["water_level", "window_size", "threshold"]
            }
            shoreline_loc = self.predict_shoreline(water_level, dune_crest, **kwargs)
            shoreline_loc = ds.interp_toe_to_grid(self.x, self.x_interp, shoreline_loc)
        elif shoreline is False or shoreline is None:
            shoreline_loc = np.full((self.z_interp.shape[0],), -1).astype(int)
        elif isinstance(shoreline, int):
            shoreline_loc = np.full((self.z_interp.shape[0],), shoreline).astype(int)
        elif len(shoreline) == self.z_interp.shape[0] & isinstance(
            shoreline, np.ndarray
        ) & all(isinstance(_, np.int64) for _ in shoreline):
            shoreline_loc = shoreline
            shoreline_loc = ds.interp_toe_to_grid(self.x, self.x_interp, shoreline_loc)
        else:
            raise ValueError(
                f"shoreline should be bool, or int (of size 1 or {self.z_interp.shape[0]})"
            )

        window = np.atleast_1d(toe_window_size)
        dt_index = np.full((self.z_interp.shape[0],), np.nan)
        for i, row in enumerate(self.z_interp):
            rr = ds.relative_relief(row, window, water_level)
            try:
                # suppress warnings for use of < with nan values
                with np.errstate(invalid="ignore"):
                    dt_index[i] = (
                        np.where(
                            rr[dune_crest_loc[i] : shoreline_loc[i]] < toe_threshold
                        )[0][-1]
                        + dune_crest_loc[i]
                    )
            except:
                dt_index[i] = (
                    np.nanargmin(rr[dune_crest_loc[i] : shoreline_loc[i]])
                    + dune_crest_loc[i]
                )
                if verbose:
                    print(
                        f"Threshold not exceeded for profile {i}, setting dune toe to minimum relief."
                    )
        dt_index = ds.interp_toe_to_grid(self.x_interp, self.x, dt_index.astype(int))
        return dt_index

    def predict_dunecrest(
        self, method="max", window_size=21, threshold=0.8, water_level=0
    ):
        """
        Find location of dune crest.

        ...

        Parameters
        ----------
        method : {'max', 'rr'}, default 'max'
            Method to identify the dune crest location. The region of the beach profile
            that the dune toe location is searched for is constrained to the region
            seaward of the dune crest.
            max: the maximum elevation of the cross-shore profile.
            rr: dune crest calculated based on relative relief.
        window_size : int, default 21
            Only valid for method "rr". Size of window for calculating relative relief.
            May be int or list of ints. If a list is passed, relative relief is
            calculated for each window size and averaged.
        threshold : float, default 0.8
            Only valid for method "rr". Threshold of relative relief that identifies the
            dune toe. Between 0 and 1.
        water_level : number, default 0
            Only valid for method "rr". Elevation of mean water level, profile elevations
            at or below mean water level elevation are set to NaN to ensure stability of
            relative relief calculations.

        Returns
        -------
        dt_index : array of ints
            dune toe location.

        """
        if method == "max":
            dc_index = np.array([np.argmax(row) for row in self.z_interp])
        elif method == "rr":
            if isinstance(window_size, int):
                assert (
                    isinstance(window_size, int)
                    & (window_size > 0)
                    & (window_size < self.z_interp.shape[1])
                ), f"window_size must be int between 0 and {self.z_interp.shape[1]}."
            elif isinstance(window_size, list):
                assert all(
                    isinstance(_, int) & (_ > 0) & (_ < self.z_interp.shape[1])
                    for _ in window_size
                ), f"window_size must be int between 0 and {self.z_interp.shape[1]}."
            else:
                raise ValueError(f"window_size must be of type int or list.")
            assert isinstance(threshold, (int, float)) & (
                0 < threshold < 1
            ), f"threshold should be number between 0 and 1, but {threshold} was passed."
            assert isinstance(
                water_level, (int, float)
            ), f"water_level should be a number, but {water_level} was passed."
            window_size = np.atleast_1d(window_size)
            dc_index = np.full((self.z_interp.shape[0],), np.nan)
            for i, row in enumerate(self.z_interp):
                rr = ds.relative_relief(row, window_size, water_level)
                try:
                    # suppress warnings for use of < with nan values
                    with np.errstate(invalid="ignore"):
                        dc_index[i] = np.where(rr > threshold)[0][-1]
                except:
                    dc_index[i] = np.nanargmin(rr)
                    print(
                        f"Threshold not found for index {i}, setting dune toe to maximum relief."
                    )
        else:
            raise ValueError(f'method should be "max" or "rr", not {method}.')

        # Interp back to original x coordinates
        dc_index = ds.interp_toe_to_grid(self.x_interp, self.x, dc_index.astype(int))
        return dc_index

    def predict_shoreline(self, water_level=0, dune_crest="max", **kwargs):
        """
        Find location of the shoreline.

        ...

        Parameters
        ----------
        water_level : number or None, default 0
            Elevation of mean water level, profile elevations at or below mean water
            level elevation are set to NaN to ensure stability of relative relief
            calculations.  See Wernette et al. (2016) for further details.
        dune_crest : {'max', 'rr', int, None}, default 'max'
            Method to identify the dune crest location. The region of the beach profile
            that the dune toe location is searched for is constrained to the region
            seaward of the dune crest.
            max: the maximum elevation of the cross-shore profile.
            rr: dune crest calculated based on relative relief.
            int: integer specifying the location of the dune crest. Of size 1 or
                 self.z.shape[0].
            None: do not calculate a dune crest location. Search the whole profile for
                  the dune toe.
        **kwargs : arguments
            Additional arguments to pass to `predict_dunecrest()`. Keywords include
            window_size, threshold, water_level.

        Returns
        -------
        numpy.ndarray
            1-D array containing the index of the shoreline location for each profile.

        """
        assert isinstance(
            water_level, (int, float)
        ), f"water_level should be a number, but {water_level} was passed."
        if dune_crest in ["max", "rr"]:
            for k in kwargs.keys():
                if k not in ["window", "threshold", "water_level"]:
                    raise Warning(
                        f"{k} not a valid argument for predict_dunecrest() or predict_shoreline()"
                    )
            kwargs = {
                k: v
                for k, v in kwargs.items()
                if k in ["window", "threshold", "water_level"]
            }
            dune_crest_loc = self.predict_dunecrest(method=dune_crest, **kwargs)
        elif isinstance(dune_crest, int):
            dune_crest_loc = np.full((self.z_interp.shape[0],), dune_crest)
        elif dune_crest is None:
            dune_crest_loc = np.full((self.z_interp.shape[0],), 0)
        elif len(dune_crest) == self.z_interp.shape[0] & isinstance(
            dune_crest, np.ndarray
        ) & all(isinstance(_, np.int64) for _ in dune_crest):
            dune_crest_loc = dune_crest
        else:
            raise ValueError(
                f'dune_crest should be "max", "rr", int (of size 1 or {self.z_interp.shape[0]}), or None'
            )

        sl_index = np.array(
            [
                len(row) - np.argmin(np.flipud(row) <= water_level) - 1
                for row in self.z_interp
            ]
        )
        # Interp back to original x coordinates
        sl_index = ds.interp_toe_to_grid(self.x_interp, self.x, sl_index)
        mask = (dune_crest_loc - sl_index) > 0
        sl_index[mask] = -1

        return sl_index
