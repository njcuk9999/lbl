#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The weights of an exposure refitted at correction time are the fit's: the
bottom of a deep line keeps its weight when its SNR is enough (it lost it to
an old depth threshold, and mask: exposure then blanked it in every
corrected file).

Created on 2026-10-23

@author: artigau
"""
import numpy as np

from lbl.pca2d import reconstruct


def test_a_deep_line_keeps_its_weight_where_its_snr_is_enough():
    n = 400
    values = np.zeros((2, n))
    values[:, 190:210] = -1.5            # a line at 22% of its continuum
    sigma = np.full((2, n), 0.02)        # SNR 50 in the log flux
    sigma[:, 300:310] = 0.5              # SNR 2: too noisy to weigh
    good = np.ones((2, n), dtype=bool)
    w = reconstruct.refit_weights(values, good, sigma, None,
                                  dict(min_pixel_snr=5.0))
    assert np.all(w[:, 190:210] > 0)
    assert np.all(w[:, 300:310] == 0)
    # the old depth threshold, only when a configuration asks for it
    old = reconstruct.refit_weights(values, good, sigma, None,
                                    dict(min_pixel_snr=None, ln_clip_low=-0.5))
    assert np.all(old[:, 190:210] == 0)
