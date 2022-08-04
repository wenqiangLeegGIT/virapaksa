#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np

def _ndvi(nir_arr,red_arr,harmonize=True):
    _ndvi_arr = np.zeros_like(nir_arr,dtype=np.float32)
    nir_arr += 1e-5
    red_arr += 1e-5
    _ndvi_arr = (nir_arr - red_arr) / (nir_arr + red_arr)
    if harmonize:
        _ndvi_arr[_ndvi_arr<0] = 0
        _ndvi_arr[_ndvi_arr>1] = 1
    return _ndvi_arr

