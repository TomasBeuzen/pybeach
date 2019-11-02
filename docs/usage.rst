Usage
-----

First import the ``Profile`` class of **pydune**:

.. code:: python

    from pydune import Profile

Given an array of cross-shore coordinates, __*x*__ of shape (*m*,) and corresponding elevations __*z*__ of shape (*m*,) for a single profile or shape (*m*, *n*) for *n* profiles, `pydune` can be used as follows to make predictions of the dune toe location:

```python

# Instantiate Profile class
p = Profile(x = x, z = z, seaward_direction = "right")

# Make dune toe predictions
toe_ml = p.predict_dunetoe_ml() # use the machine learning (ML) method
toe_mc = p.predict_dunetoe_mc() # use the maximum curvature (MC) method
toe_rr = p.predict_dunetoe_rr() # use the relative relief (RR) method
toe_pd = p.predict_dunetoe_pd() # use the perpendicular distance (PD) method

```