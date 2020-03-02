Usage
-----

First import the ``Profile`` class of **pybeach**:

.. code:: python

    from pybeach import Profile

Given an array of cross-shore coordinates, *x* of shape (*m*,) and corresponding elevations *z* of shape (*m*,) for a single profile or shape (*m*, *n*) for *n* profiles, `pybeach` can be used as follows to make predictions of the dune toe location:

.. code:: python

    # example data
    import numpy as np
    x = np.arange(0, 80, 0.5)
    z = np.concatenate((np.linspace(4, 5, 40),
                        np.linspace(5, 2, 10),
                        np.linspace(2, 0, 91)[1:],
                        np.linspace(0, -1, 20)))

    # Instantiate Profile class
    p = Profile(x, z)

    # Predict dune toe location
    toe_ml = p.predict_dunetoe_ml('mixed_clf') # use the machine learning (ML) method
    toe_mc = p.predict_dunetoe_mc() # use the maximum curvature (MC) method
    toe_rr = p.predict_dunetoe_rr() # use the relative relief (RR) method
    toe_pd = p.predict_dunetoe_pd() # use the perpendicular distance (PD) method

    # (optional) Predict shoreline and dune crest location
    crest = p.predict_dunecrest()
    shoreline = p.predict_shoreline()