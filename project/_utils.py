import skimage as skim
import sklearn 
import cv2
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import napari

def median_bin_rgb(image, block_size=(2, 2)):
    channels = []
    for i in range(3):  # R, G, B
        channel = skim.measure.block_reduce(image[:, :, i], block_size=block_size, func=np.median)
        channels.append(channel)
    return np.stack(channels, axis=-1).astype(np.uint8)

def hsv_spx_feat(hsv_img, spx, agg_func=np.median):
    h = np.zeros(np.max(spx)+1)
    s = np.zeros(np.max(spx)+1)
    v = np.zeros(np.max(spx)+1)
    for i in range(np.max(spx)+1):
        mask = spx == i
        h[i] = agg_func(hsv_img[mask,0])
        s[i] = agg_func(hsv_img[mask,1])
        v[i] = agg_func(hsv_img[mask, 2])

    hx = np.cos(h/255*2*np.pi)
    hy = np.sin(h/255*2*np.pi)

    feat = np.array([hx, hy, s, v])
    X = sklearn.preprocessing.StandardScaler().fit_transform(feat.T)

    return X

def rgb_spx_feat(rgb_img, spx, agg_func=np.median):
    for i in range(np.max(spx)+1):
        mask = spx == i
        r = agg_func(rgb_img[mask,0])
        g = agg_func(rgb_img[mask,1])
        b = agg_func(rgb_img[mask, 2])

    feat = np.array([r, g, b])
    X = sklearn.preprocessing.StandardScaler().fit_transform(feat.T)

    return X

def gmm_on_spx_fit(c_range, X):
    best_bic = np.inf
    best_gmm = None
    best_n_components = 0
    bics = []

    for c in c_range:
        gmm = sklearn.mixture.GaussianMixture(n_components=c, covariance_type='full', reg_covar=1e-3)
        gmm.fit(X)
        bic = gmm.bic(X)
        bics.append(bic)
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
            best_n_components = c

    return best_gmm

def gmm_on_spx_predict(gmm, X, spx):
    labels = gmm.predict(X.reshape((-1, X.shape[2])))
    labels = labels.reshape((spx.shape[0], spx.shape[1]))
    return labels