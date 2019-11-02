[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://travis-ci.org/TomasBeuzen/pydune.svg?branch=master)](https://travis-ci.org/TomasBeuzen/pydune)
[![Documentation Status](https://readthedocs.org/projects/pydune-tomasbeuzen/badge/?version=latest)](https://pydune-tomasbeuzen.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/TomasBeuzen/pydune/badge.svg?branch=master)](https://coveralls.io/github/TomasBeuzen/pydune?branch=master)

# **pydune**: A Python package for locating the dune toe on cross-shore beach profile transects.

<div align="center">
  <img src="docs/img/figure_1.png" alt="pydune-example" width="700"/>
</div>

## Background
**pydune** is a python package for identifying dune toes on 2D beach profile transects. It includes the following methods:
  - Machine learning; 
  - Maximum curvature (Stockdon et al, 2007); 
  - Relative relief (Wernette et al, 2016); and,
  - Perpendicular distance.
  
In addition, **pydune** contains methods for identifying the shoreline position and dune crest position on 2D beach profile transects. See the [*pydune* paper](paper.md) for more details about **pydune**.

## Usage
```python
from pydune import Profile

# example data
x = np.arange(0, 80, 0.5)
z = np.concatenate((np.linspace(4, 5, 40),
                    np.linspace(5, 2, 10),
                    np.linspace(2, 0, 91)[1:],
                    np.linspace(0, -1, 20)))

# instantiate
p = Profile(x, z)

# predict dune toe, dune crest, shoreline location
toe_ml, prob_ml = p.predict_dunetoe_ml('wave_embayed_clf')  # predict toe using machine learning model
toe_mc = p.predict_dunetoe_mc()    # predict toe using maximum curvature method (Stockdon et al, 2007)
toe_rr = p.predict_dunetoe_rr()    # predict toe using relative relief method (Wernette et al, 2016)
toe_pd = p.predict_dunetoe_pd()    # predict toe using perpendicular distance method
crest = p.predict_dunecrest()      # predict dune crest
shoreline = p.predict_shoreline()  # predict shoreline
```

See the [example notebook](https://github.com/TomasBeuzen/pydune/blob/master/example.ipynb) for more details.

## Documentation
Read the **pydune** documentation [here](https://pydune-tomasbeuzen.readthedocs.io/en/latest/?badge=latest).

## Questions, Comments, Suggestions
Do you have a question that needs answering? Have you found an issue with the code and need to get it fixed? Or perhaps you're looking to contribute to the code and have ideas for how it could be improved. In all cases, please see the [Issues](https://github.com/TomasBeuzen/pydune/issues) page.

## References
Stockdon, H. F., Sallenger Jr, A. H., Holman, R. A., & Howd, P. A. (2007). A simple model for the spatially-variable coastal response to hurricanes. Marine Geology, 238, 1-20. https://doi.org/10.1016/j.margeo.2006.11.004

Wernette, P., Houser, C., & Bishop, M. P. (2016). An automated approach for extracting Barrier Island morphology from digital elevation models. Geomorphology, 262, 1-7. https://doi.org/10.1016/j.geomorph.2016.02.024
