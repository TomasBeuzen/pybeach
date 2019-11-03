# -*- coding: utf-8 -*-
"""
This is the unittest for pydune.

TO DO:
- define a test data fixture
- use it in the fails especially
"""

import numpy as np
from pytest import approx, raises, fixture
from pydune.pydune import Profile


@fixture()
def models():
    x = np.arange(0, 80, 0.5)
    z = np.hstack((np.linspace(4, 5, 40),
                   np.linspace(5, 2, 10),
                   np.linspace(2, 0, 91)[1:],
                   np.zeros((20,))))

    toe = np.array([51])
    crest = np.array([38])
    shoreline = np.array([140])
    pydune1d = Profile(x, z)
    pydune2d = Profile(x, np.vstack((z, z)))
    return pydune1d, pydune2d, toe, crest, shoreline

@fixture()
def data():
    x = np.arange(0, 80, 0.5)
    z1d = np.hstack((np.linspace(4, 5, 40),
                     np.linspace(5, 2, 10),
                     np.linspace(2, 0, 91)[1:],
                     np.zeros((20,))))
    z2d = np.vstack((z1d, z1d))
    return x, z1d, z2d

class Testpydune(object):

    def test_predict_dunetoe_ml(self, models):
        pydune1d, pydune2d, toe, crest, shoreline = models
        assert pydune1d.predict_dunetoe_ml('barrier_island_clf')[0] == approx(toe, abs=10)
        assert pydune1d.predict_dunetoe_ml('wave_embayed_clf')[0] == approx(toe, abs=10)
        assert pydune1d.predict_dunetoe_ml('mixed_clf')[0] == approx(toe, abs=10)
        assert pydune1d.predict_dunetoe_ml('mixed_clf', dune_crest=40)[0] == approx(toe, abs=10)
        assert pydune1d.predict_dunetoe_ml('mixed_clf', dune_crest=None)[0] == approx(toe, abs=10)
        assert pydune1d.predict_dunetoe_ml('mixed_clf', dune_crest=np.array([40]))[0] == approx(toe, abs=10)
        # assert pydune2d.predict_dunetoe_ml('SR04_clf')[0] == approx(np.hstack((toe, toe)), abs=10)

    def test_predict_dunetoe_mc(self, models):
        pydune1d, pydune2d, toe, crest, shoreline = models
        assert pydune1d.predict_dunetoe_mc(dune_crest='max') == approx(toe, abs=10)
        assert pydune1d.predict_dunetoe_mc(dune_crest='max', hanning_window=3) == approx(toe, abs=10)
        assert pydune1d.predict_dunetoe_mc(dune_crest=40)[0] == approx(toe, abs=10)
        assert pydune1d.predict_dunetoe_mc(dune_crest=None)[0] == approx(toe, abs=10)
        assert pydune1d.predict_dunetoe_mc(dune_crest=np.array([40]))[0] == approx(toe, abs=10)
        assert pydune1d.predict_dunetoe_mc(shoreline=False) == approx(toe, abs=10)
        assert pydune1d.predict_dunetoe_mc(shoreline=159) == approx(toe, abs=10)
        assert pydune1d.predict_dunetoe_mc(shoreline=np.array([159])) == approx(toe, abs=10)
        assert pydune2d.predict_dunetoe_mc(dune_crest='rr') == approx(np.hstack((toe, toe)), abs=10)
        assert pydune1d.predict_dunetoe_mc(dune_crest='max') == approx(toe, abs=10)
        assert pydune2d.predict_dunetoe_mc(dune_crest='rr') == approx(np.hstack((toe, toe)), abs=10)

    def test_predict_dunetoe_rr(self, models):
        pydune1d, pydune2d, toe, crest, shoreline = models
        assert pydune1d.predict_dunetoe_rr() == approx(toe, abs=10)
        assert pydune2d.predict_dunetoe_rr() == approx(np.hstack((toe, toe)), abs=10)

    def test_predict_dunetoe_pd(self, models):
        pydune1d, pydune2d, toe, crest, shoreline = models
        assert pydune1d.predict_dunetoe_pd(dune_crest='max') == approx(toe, abs=10)
        assert pydune1d.predict_dunetoe_pd(dune_crest=40)[0] == approx(toe, abs=10)
        assert pydune1d.predict_dunetoe_pd(dune_crest=None)[0] == approx(toe, abs=10)
        assert pydune1d.predict_dunetoe_pd(dune_crest=np.array([40]))[0] == approx(toe, abs=10)
        assert pydune1d.predict_dunetoe_pd(shoreline=False) == approx(toe, abs=10)
        assert pydune1d.predict_dunetoe_pd(shoreline=159) == approx(toe, abs=10)
        assert pydune1d.predict_dunetoe_pd(shoreline=np.array([159])) == approx(toe, abs=10)
        assert pydune2d.predict_dunetoe_pd(dune_crest='rr') == approx(np.hstack((toe, toe)), abs=10)

    def test_predict_dunecrest(self, models):
        pydune1d, pydune2d, toe, crest, shoreline = models
        assert pydune1d.predict_dunecrest(method='max') == approx(crest, abs=10)
        assert pydune2d.predict_dunecrest(method='rr') == approx(np.hstack((crest, crest)), abs=10)

    def test_predict_shoreline(self, models):
        pydune1d, pydune2d, toe, crest, shoreline = models
        assert pydune1d.predict_shoreline(dune_crest='max') == approx(shoreline, abs=10)
        assert pydune1d.predict_shoreline(dune_crest=40)[0] == approx(shoreline, abs=10)
        assert pydune1d.predict_shoreline(dune_crest=None)[0] == approx(shoreline, abs=10)
        assert pydune1d.predict_shoreline(dune_crest=np.array([40]))[0] == approx(shoreline, abs=10)
        assert pydune2d.predict_shoreline(dune_crest='rr') == approx(np.hstack((shoreline, shoreline)), abs=10)


class TestpyduneFails(object):

    def test_bad_input(self, data):
        x, z1d, z2d = data
        with raises(TypeError):  # no input
            Profile()
        with raises(TypeError):  # only one input
            Profile(x)
        with raises(AssertionError):  # list input
            Profile(list(x), z1d)
        with raises(AssertionError):  # list input
            Profile(x, list(z1d))
        with raises(AssertionError):  # string input
            Profile('x', z1d)
        with raises(AssertionError):  # multidimensional x
            Profile(z2d, x)
        with raises(ValueError):  # x and z don't share dimension
            Profile(np.arange(10), z1d)
        with raises(Warning):  # profiles with wrong orientation (sea on left)
            Profile(x, np.flipud(z1d))

    def test_bad_method_calls(self, models):
        pydune1d, _, _, _, _ = models
        with raises(Warning):
            pydune1d.predict_dunetoe_ml('wave_embayed_clf', bad_key_word=123)
            pydune1d.predict_dunetoe_mc(bad_key_word=123)
            pydune1d.predict_dunetoe_rr(bad_key_word=123)
            pydune1d.predict_dunetoe_pd(bad_key_word=123)
            pydune1d.predict_shoreline(bad_key_word=123)
        with raises(ValueError):  # bad method
            pydune1d.predict_dunecrest(method='m')
            pydune1d.predict_dunecrest(method=1)
            pydune1d.predict_dunetoe_ml('wave_embayed_clf', dune_crest='bad_method')
            pydune1d.predict_dunetoe_mc(shoreline='ok')
        with raises(AssertionError):
            pydune1d.predict_dunetoe_mc(window_size=-1)
            pydune1d.predict_dunetoe_mc(dune_crest='max', hanning_window=-1)
            pydune1d.predict_dunetoe_mc(dune_crest='max', hanning_window='string')
            pydune1d.predict_dunetoe_rr(window_size=-1)
            pydune1d.predict_dunetoe_rr(threshold=-1)
            pydune1d.predict_dunetoe_rr(water_level='1')
            pydune1d.predict_dunetoe_ml(1)
            pydune1d.predict_dunetoe_ml('SR04_clf', -1)
            pydune1d.predict_dunecrest(method="rr", threshold=1.1)
            pydune1d.predict_dunecrest(method="rr", threshold=-0.1)
            pydune1d.predict_dunecrest(window_size=-1)
        with raises(FileNotFoundError):
            pydune1d.predict_dunetoe_ml('bad_file_name')

