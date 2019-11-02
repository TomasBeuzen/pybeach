Background
----------

Dunes along sandy coastlines provide an important natural barrier to coastal hazards such as storm-induced waves and surge. The capacity of sandy dunes to provide coastal hazard protection depends in large part on their geometry. In particular, the location of the dune toe (the transition point between the beach and dune) is a key factor used in coastal erosion models and for assessing coastal vulnerability to hazards. **pydune** is a Python package for locating the dune toe on cross-shore beach profile transects. The aim of *pydune* is to collate commonly used algorithms for dune toe identification and to provide a new method of locating the dune toe based on machine learning.

 **pydune** currently includes the following methods for locating the dune toe on a cross-shore beach profile:
  - Machine learning;
  - Maximum curvature (Stockdon et al, 2007) [#sto07]_;
  - Relative relief (Wernette et al, 2016) [#wer16]_; and,
  - Perpendicular distance.

See the [*pydune* paper](paper.md) for more details of these methods.

.. image:: ./_static/figure_1.png
..

    | Example application of **pydune**.

--------

.. [#vit17] Wernette, P., Houser, C., & Bishop, M. P. "An automated approach        for extracting Barrier Island morphology from digital elevation models."        Geomorphology, 262 (2016): 1-7. https://doi.org/10.1016/j.geomorph.2016.02.024.
.. [#sto06] Stockdon, H. F., Sallenger Jr, A. H., Holman, R. A., & Howd, P. A.      "A simple model for the spatially-variable coastal response to hurricanes."     Marine Geology, 238 (2007): 1-20. https://doi.org/10.1016/j.margeo.2006.11.004