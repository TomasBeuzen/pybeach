# **pydune**: A Python package for locating the dune toe on cross-shore beach profile transects.

<div align="center">
  <img src="docs/figure_1.png" alt="pydune-example" width="700"/>
</div>

## What is it?
**pydune** is a python package for identifying dune toes on beach profile transects. It includes the following methods:
  - Maximum curvature
  - Relative relief
  - Perpendicular distance
  - Machine learning

See paper for more info.

## Usage
```sh
from pydune import profile

p = profile(x, z, "right")
p.dt_con(no_of_outputs=1)
p.dt_si(no_of_outputs=1)
p.dt_lcp(no_of_outputs=1)
p.dt_rr(no_of_outputs=1)
p.dt_ml(no_of_outputs=1, model_type="barrier_island")
```
