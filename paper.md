---
title: 'pydune: A Python module for extracting the location of dune toes on beach profile transects'
tags:
  - Python
  - coastal
  - morphology
  - dunes
  - machine learning
authors:
  - name: Tomas Beuzen
    orcid: 0000-0003-2762-9151
    affiliation: 1
affiliations:
 - name: Department of Statistics, University of British Columbia, Vancouver, Canada
   index: 1
date: 04 October 2019
bibliography: paper.bib
---

# Summary

Dunes along sandy coastlines provide an important natural barrier to coastal hazards. The capacity
of sandy dunes to provide coastal hazard protection depends in large part on their geometry. In
particular, the location of the dune toe (the transition point between the beach and dune) is a key
factor used in coastal erosion models and management decisions. There are many different algorithms
currently available for defining the dune toe. The *pydune* package documented herein is motivated
by two key aspects: 1. to collect existing done toe location algorithms in a single, functional Python
package; and, 2. to provide an additional new method for identifying dune toe location based on machine
learning.

*pydune* is an open-source Python package that allows a user to quickly and effectively identify the
dune toe location on beach profiles using existing techniques or a new data-driven, ML-based
technique. The user inputs into *pydune* an array of cross-shore coordinates of shape (*m*,)  and an
array of corresponding elevations of shape (*m*,) for a single beach profile or shape (*m*, *n*) for *n*
profiles. The user can then use *pydune* to identify the location of the dune toe using the
following methods:

1. maximum curvature [@Stockdon2007];
2. relative relief [@Wernette2016];
3. perpendicular distance; and,
4. a machine learning model.

Figure 1 shows an example of *pydune* applied to a beach profile transect. The machine learning
approach to identify the location of the dune toe is a random forest classifier trained on 1500
beach profile transects for which the dune toe has been expertly identified. This approach is novel,
and when tested on 200 unseen profiles, outperformed the other methods with a mean absolute error
(MAE) of XXX, compared to XXX (maximum curvature), YYY (relative relief), ZZZ (perpendicular
distance). Importantly, the machine learning methodology used to create the dune toe model
(described in detail below) can be applied more generally to the identification of topographic
features in other coastal applications (such as dune crests, berm crests, wave runup) and
geophysical applications.

![pydune-example-figure](docs/paper_figures/figure_1.png)

# Statement of Need

Domain experts are generally able to identify the location of the dune toe given a 2D transect of a
beach profile. However, recent improvement in coastal monitoring technologies (such a optical remote
sensing, LIDAR, satellite) and data collection, have resulted in a significant increases of coastal
topographic data, for which analysis by an expert is infeasible. As a result, there has been
increased need for reliable and efficient algorithms for extracting features such as dune toes from
topographic coastal data. To date several different algorithms have been developed for this purpose,
including: . Over the last decade or so, MATLAB has been the primary processing environment in
coastal research, but with increased use of open-source data, software, and machine learning, Python
is becoming a more popular programming language in coastal research and practice. One aim of
*pydune* is to collect existing dune toe location algorithms into a single Python package to
facilitate future use.

A recent study by [@Wernette2018] analysing many existing approaches for extracting dune toe
locations on bech profiles found that there is considerable variation in how these algorithms define
and locate the dune toe. As a result, expert checking is often required to validate the results of
these algorithms. A key issue is that, while an expert can generally identify the dune toe on a
given beach profile transect, it is difficult to define an algorithm that can reliably define the
dune toe for the large variety of beach profiles and conditions encountered in nature. Here we
propose an alternative approach to creating a dune toe model using machine learning. In this
approach, the idea is to encode expert knowledge to create a model that is applicable to a large
variety fo beach profile shapes, efficient, and scalable such that it can be updated and imporved as
additional data becomes available in the future. This machine learning model approach is discussed
in detail below.

# pydune

The *pydune* Python module provides a *Profile* class in *pydune.py*. This class contains methods
for defining the dune toe using each of the approaches listed above. *pydune* utilises support
functions located within the *classifier_support.py* and *data_support.py* modules.

Details of the dune toe location algorithms of maximum curvature and relative relief can be found in
[@Stockdon2007] and [@Wernette2016] respectively. The perpendicular distance method is a simple
algorithm that draw a straight line between the dune crest and shoreline and calculates the dune toe
as the location on the beach profile with maximum perpendicular distance to that line.

The novel dune toe location method provided by *pydune* is the machine learning classifier. The
methodology used to create the classifier is discussed briefly below, however more details can be
found in the provided *create_classifier.py* and *classifier_support.py* modules. The dune toe
classifier provided with *pydune* was developed using 1468 individual beach profile transects
collected by airborne lidar pre- and post-Hurricane Ivan from Santa Rosa Island, Florida, US. This
data is provided courtesy of XXX and is freely available at (). Each profile was separately
inspected and the dune toe location was manually determined by an expert. As beach profile transects
can vary significantly in length (i.e., from 10’s of meters to 100’s of meters), the dune toe
classifier developed here was based on inputting a specific length of transect (referred to as a
“window”), rather than an entire profile, for which the classifier predicts the probability of the
dune toe being located at the centre of the window (Figure Xa). The elevation of beach profiles can
also vary significantly. As model inputs should be as generalized as possible, the gradient of a
profile elevations within a window (i.e., the first derivative of the profile elevation) is used as
input into the model (Figure Xb). Finally, to train the classifier requires both positive and
negative example of "windows". The negative dune toe can effectively be any point other than the
actual dune toe. However, it should not be so close to the actual dune toe as to confuse the model.
We therefore define a “buffer zone” around the actual dune toe location (Figure Xc) from which to
sample negative samples. A single negative dune toe sample was randomly extracted from each beach
profile to train the machine learning classifier, resulting in 1468 positive samples and 1468
negative samples. A random forest classifier algorithm was used to develop the models, with an
ensemble of 100 trees and no maximum depth. While multiple classifier were trialled, the random
forest classifier gave the highest accuracy. In addition, it outputs a probability of dune toe
across the profile which can be useful for expert interpretation. Nevertheless, the script
*create_classifier.py* is provided in the *pydune* GitHub repository so that users may create their
own classifiers from their own data and using their desired machine learning algorithm.

The two parameters of this approach were the window size and buffer size discussed above. A grid
search was conducted over different values of these two aprameters, and a window size of 20 m and
buffer size of 20 m was found to be optimal. See the supplemet figure X.

In developing the machine learning dune toe locator, the full dataset of 1468 profiles was initially
split into a 80% training and 20% test set. The classifier was developed on the training data and
its performance on the unseen test data was quantified and compared to the other algorithms. Figure
Y and Table Y show that the machine learning model outperformed the other classifiers on the unseen
test data.

[figure-X-test-data-box-plots]()
[table-X-test-data-results]()

# Usage

Given an array of cross-shore coordinates, __*x*__ of shape (*m*,)  and corresponding elevations
__*z*__ of shape (*m*,) for a single profile or shape (*m*, *n*) for *n* profiles, `pydune` can be
used as follows to make predictions of the dune toe location:

```python
from pydune import Profile

# Instantiate Profile class
p = Profile(x = x, z = z, seaward_direction = "right")

# Make dune toe predictions
toe_ml = p.predict_dunetoe_ml() # use the machine learning (ML) method
toe_mc = p.predict_dunetoe_mc() # use the maximum curvature (MC) method
toe_rr = p.predict_dunetoe_rr() # use the relative relief (RR) method
toe_pd = p.predict_dunetoe_pd() # use the perpendicular distance (PD) method

```

The `pydune` source code can be found on [github](). Additional documentation and examples are
provided [here]().

# Acknowledgements



# References