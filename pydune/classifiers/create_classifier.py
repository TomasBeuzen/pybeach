# -*- coding: utf-8 -*-
"""
Created on Wed Mar 6 12:43:40 2019

@author: Tomas Beuzen

Script to create SR04 dune toe ML model.
"""

import pickle
import joblib
from pydune.support import classifier_support as cs

# Load data
with open('../../data/SR04_train.pkl', 'rb') as f:
    SR04_data = pickle.load(f)
x, z, toe = SR04_data['x'], SR04_data['z'], SR04_data['toe']

# Create classifier
clf = cs.create_classifier(x, z, toe, window=40, min_buffer=40, max_buffer=200)

# Save classifier
with open("../classifiers/" + clf_name) as f:
    joblib.dump(clf, f, protocol=pickle.HIGHEST_PROTOCOL)