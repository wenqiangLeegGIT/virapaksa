#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import abc

import numpy as np
import scipy as sp
from osgeo import gdal
from sklearn.linear_model import orthogonal_mp_gram



class ApproximateKSVD(object):
    def __init__(self, n_components, max_iter=10, tol=1e-6,
                 transform_n_nonzero_coefs=None):
        """
        Parameters
        ----------
        n_components:
            Number of dictionary elements
        max_iter:
            Maximum number of iterations
        tol:
            tolerance for error
        transform_n_nonzero_coefs:
            Number of nonzero coefficients to target
        """
        self.components_ = None
        self.max_iter = max_iter
        self.tol = tol
        self.n_components = n_components
        self.transform_n_nonzero_coefs = transform_n_nonzero_coefs

    def _update_dict(self, X, D, gamma): # D:(100,70) gamma:(70,150)
        _X = np.zeros_like(X)
        _D = np.zeros_like(D)
        _gamma = np.zeros_like(gamma)
        for j in range(self.n_components):
            non_zeros = gamma[j, :] != 0
            if np.any(non_zeros):
                jd = D[:,j][:,np.newaxis] @ gamma[j,non_zeros][np.newaxis,:]
                _X[:,non_zeros] += jd
                err = X - _X
                err_j = err[:,non_zeros]
                u,s,v = np.linalg.svd(err_j)
                _D[:,j] = u[:,0]
                _gamma[j,non_zeros] = v[:,0] * s[0]
        return _D,_gamma


    def _initialize(self, X):
        if min(X.shape) < self.n_components:
            D = np.random.randn(self.n_components, X.shape[1])
        else:
            u, s, vt = sp.sparse.linalg.svds(X, k=self.n_components)
            D = np.dot(np.diag(s), vt)
        D /= np.linalg.norm(D, axis=1)[:, np.newaxis]
        # return D
        return np.random.randn(X.shape[0],self.n_components)

    def _transform(self, D, X):
        # gram = D.dot(D.T)
        # Xy = D.dot(X.T)
        gram = D.T @ D
        Xy = D.T @ X

        n_nonzero_coefs = self.transform_n_nonzero_coefs
        if n_nonzero_coefs is None:
            n_nonzero_coefs = int(0.1 * X.shape[1])

        return orthogonal_mp_gram(
            gram, Xy, n_nonzero_coefs=n_nonzero_coefs)

    def fit(self, X):
        """
        Parameters
        ----------
        X: shape = [n_samples, n_features]
        """
        D = self._initialize(X)
        for i in range(self.max_iter):
            gamma = self._transform(D, X)
            # e = np.linalg.norm(X - gamma.dot(D))
            e = np.linalg.norm(X - D.dot(gamma))
            if e < self.tol:
                break
            D, gamma = self._update_dict(X, D, gamma)

        self.components_ = D
        return self

    def transform(self, X):
        return self._transform(self.components_, X)

def img2arr(img,dtype=np.float32):
    arr = gdal.Open(img)
    out = np.zeros((arr.RasterCount,arr.RasterXSize*arr.RasterYSize))
    for i in range(arr.RasterCount):
        b = arr.GetRasterBand(i+1)
        out[i,:] = b.ReadAsArray().ravel()

    out = out.astype(dtype=dtype)
    del arr
    return out.T

def arr2img(arr,*,ref=None,xsize=None,ysize=None,inter="bsq"):
    assert ref is not None or (xsize is not None and ysize is not None),"must provide ref or (rows and cols)"
    if ref is not None:
        ds = gdal.Open(ref)
        xsize = ds.RasterXSize
        ysize = ds.RasterYSize

    out = np.zeros(arr.shape[0],ysize,xsize)
    for i in range(arr.shape[0]):
        out[i] = arr[i].reshape(ysize,xsize)

    if inter == "bil":
        out = np.transpose(out,[1,2,0])
    return out



if __name__ == '__main__':
    # from PIL import Image
    # from osgeo import gdal
    sample = np.random.rand(100,150)
    # cat = r"C:\Users\pc019\Pictures\96a58e0fa08f03e53c6be530623753dd_Z.jpg"
    # img = Image.open(cat)
    # imgarr = img2arr(cat)
    # imgarr = (imgarr - imgarr.mean()) / np.std(imgarr)
    # ksvd = ApproximateKSVD(120,max_iter=30)
    # ksvd.fit(imgarr)
    # gamma = ksvd.transform(imgarr)
    # D = ksvd.components_
    # recons = (D @ gamma).flatten()
    # recons = np.interp(recons,(recons.min(),recons.max()),(0,255)).astype(np.uint8).reshape(b.height,b.width)
    # reconsimg = Image.fromarray(recons).convert("RGB")
    # reconsimg.save(r"C:\Users\pc019\Pictures\96a58e0fa08f03e53c6be530623753dd_Z_0.jpg")
    # print('')
    import utm
    x = utm.from_latlon(32.125,141.2)
    print(x)