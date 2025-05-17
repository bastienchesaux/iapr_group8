import skimage as skim
import sklearn 
import cv2
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import napari

from skimage.measure import regionprops

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
        h[i] = agg_func(hsv_img[mask,0])/180
        s[i] = agg_func(hsv_img[mask,1])/255
        v[i] = agg_func(hsv_img[mask, 2])/255

    hx = np.cos(h*2*np.pi)
    hy = np.sin(h*2*np.pi)

    feat = np.array([h, s, v])
    X = sklearn.preprocessing.StandardScaler().fit_transform(feat.T)

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

def features_objects(contours, binned, processed):
    # initialize lists to store images of each object
    # and features
    count_pixel = []
    diff_object = []

    f_peri = []
    f_area = []
    f_comp = []
    f_rect = []

    i = 0
    for contour in contours:
        count_pixel.append(0)
        masque = np.zeros_like(processed)
        cv2.drawContours(masque, [contour], -1, 255, thickness= cv2.FILLED)
        chocolate = np.zeros_like(binned)
        N, M, _ = binned.shape
        for x in range(N):
            for y in range(M):
                if masque[x, y] != 0:
                    chocolate[x, y, :] = binned[x, y, :]
                    count_pixel[i] += 1
        diff_object.append(chocolate)
        # compute features
        properties = regionprops(label_image=masque)
        f_peri.append(properties[0].perimeter)
        f_area.append(properties[0].area)
        f_comp.append(f_peri[i]**2/f_area[i])
        f_rect.append(f_area[i]/properties[0].area_bbox)
        i += 1
    return diff_object, count_pixel, f_peri, f_area, f_comp, f_rect