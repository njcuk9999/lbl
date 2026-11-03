#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The PHOENIX models come by HTTPS: the Goettingen FTP path is replaced by its
HTTPS path under data/, which redirects to the GWDG mirror.

Created on 2026-11-02

@author: artigau
"""
import pytest

from lbl.core import base_classes, parameters
from lbl.instruments import select


def _registered():
    for instrument, sources in select.InstDict.items():
        if instrument == 'Generic':
            continue
        for source, klass in sources.items():
            yield instrument, source, klass


@pytest.mark.parametrize('instrument,source,klass', list(_registered()))
def test_stellar_model_urls_are_https(instrument, source, klass):
    inst = klass(parameters.params.copy(), base_classes.ParamDict())
    for key in ('STELLAR_WAVE_URL', 'STELLAR_MODEL_URL'):
        url = inst.params[key]
        assert url.startswith('https://phoenix.astro.physik.uni-goettingen.de/'
                              'data/HiResFITS/'), (key, url)
