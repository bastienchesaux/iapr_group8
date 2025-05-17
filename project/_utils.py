import skimage as skim
import sklearn 
import cv2
import numpy as np
import scipy.ndimage as ndi

import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import napari
from typing import Callable

def median_bin_rgb(image, block_size=(2, 2)):
    channels = []
    for i in range(3):  # R, G, B
        channel = skim.measure.block_reduce(image[:, :, i], block_size=block_size, func=np.median)
        channels.append(channel)
    return np.stack(channels, axis=-1).astype(np.uint8)

def hsv_spx_feat(hsv_img, spx, agg_func=np.median):    
    label_ids = np.arange(spx.max() + 1)        
    h = ndi.labeled_comprehension(hsv_img[..., 0],labels=spx,index=label_ids,func=agg_func,out_dtype=float,default=np.nan)/180 #cv2 convention
    s = ndi.labeled_comprehension(hsv_img[..., 1],spx,label_ids,func=agg_func,out_dtype=float,default=np.nan)/255
    v = ndi.labeled_comprehension(hsv_img[..., 2],spx,label_ids,func=agg_func, out_dtype=float,default=np.nan)/255

    hx = np.cos(h*2*np.pi)
    hy = np.sin(h*2*np.pi)



    feat = np.array([hx, hy, s, v])
    return feat.T

def rgb_spx_feat(rgb_img, spx, agg_func=np.median):
    for i in range(np.max(spx)+1):
        mask = spx == i
        r = agg_func(rgb_img[mask,0])
        g = agg_func(rgb_img[mask,1])
        b = agg_func(rgb_img[mask, 2])

    feat = np.array([r, g, b])
    X = sklearn.preprocessing.StandardScaler().fit_transform(feat.T)

    return X

def gmm_on_spx_fit(c_range, X, plot=False):
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
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(c_range, bics, marker='o')
        plt.title('BIC vs Number of Components')
        plt.tight_layout()
        plt.show()  

    return best_gmm

def gmm_on_spx_predict(gmm, X, spx):
    labels = gmm.predict(X.reshape((-1, X.shape[2])))
    labels = labels.reshape((spx.shape[0], spx.shape[1]))
    return labels



def kl_divergence_gaussians(mu0, cov0, mu1, cov1):
    k = len(mu0)
    cov1_inv = np.linalg.inv(cov1)
    diff = mu1 - mu0

    term1 = np.trace(cov1_inv @ cov0)
    term2 = 5*diff.T @ cov1_inv @ diff
    term3 = np.log(np.linalg.det(cov1) / np.linalg.det(cov0))
    return 0.5 * (term1 + term2 - k + term3)

def symmetric_kl(mu0, cov0, mu1, cov1):
    return 0.5 * (
        kl_divergence_gaussians(mu0, cov0, mu1, cov1) +
        kl_divergence_gaussians(mu1, cov1, mu0, cov0)
    )


def merge_labels(labels, target, merge_ratio):
    merged = labels.copy()

    mask = labels==target
    boundary = ndi.binary_dilation(mask, np.ones((3,3))) ^ mask
    touching = np.unique(labels[boundary])

    for other in touching:
        ratio = np.count_nonzero(labels[boundary]==other)/min(np.count_nonzero(mask), np.count_nonzero(labels==other))
        print(f'ratio of contact between {target} and {other}: {ratio}')
        if ratio > merge_ratio:
            merged[labels==other] = target
    
    return merged