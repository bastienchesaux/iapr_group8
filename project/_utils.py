import skimage as skim
import sklearn 
import scipy
import cv2
import numpy as np
import scipy.ndimage as ndi

import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import napari
from typing import Callable

from skimage.measure import regionprops

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
        #print(f'ratio of contact between {target} and {other}: {ratio}')
        if ratio > merge_ratio:
            merged[labels==other] = target
    
    return merged
def rgb_to_hsv(rgb_img):
    bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    return hsv_img

def mean_from_mask(img, mask):
    mean = np.mean(img[mask])
    return mean

def offset_range(img, mean, offset):
    img_range = ((img < mean + offset) & (img > mean - offset))
    return img_range

def region_growing(rgb_img, hsv_img, mask):
    mask = skim.morphology.binary_closing(mask, skim.morphology.disk(5))
    mask = skim.morphology.remove_small_holes(mask, area_threshold=200)
    mask = skim.morphology.remove_small_objects(mask, min_size=50)
    mask = mask.astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rg_mask = np.zeros_like(mask)
    rg_mask = rg_mask.astype(bool)

    for contour in contours:

        small_mask = np.zeros_like(mask)

        cv2.drawContours(small_mask, [contour], -1, 255, thickness=cv2.FILLED)

        small_mask = small_mask.astype(bool)

        mean_red = mean_from_mask(rgb_img[:,:,0], small_mask)
        mean_green = mean_from_mask(rgb_img[:,:,1], small_mask)
        mean_blue = mean_from_mask(rgb_img[:,:,2], small_mask)

        mean_hue = mean_from_mask(hsv_img[:,:,0], small_mask)
        mean_sat = mean_from_mask(hsv_img[:,:,1], small_mask)
        mean_val = mean_from_mask(hsv_img[:,:,2], small_mask)

        offset_hue = 54
        offset_sat = 40
        offset_val = 40

        rgb_offset = 30
        offset_red = rgb_offset
        offset_green = rgb_offset
        offset_blue = rgb_offset
        
        labelled = (offset_range(hsv_img[:,:,0], mean_hue, offset_hue) & offset_range(hsv_img[:,:,1], mean_sat, offset_sat) & offset_range(hsv_img[:,:,2], mean_val, offset_val) &
                    offset_range(rgb_img[:,:,0], mean_red, offset_red) & offset_range(rgb_img[:,:,1], mean_green, offset_green) & offset_range(rgb_img[:,:,2], mean_blue, offset_blue))

        small_mask = skim.morphology.binary_erosion(small_mask, skim.morphology.disk(5))

        n_iterations = 10
        
        for i in range(n_iterations):         
            small_mask |= (skim.morphology.binary_dilation(small_mask, skim.morphology.disk(3)) & labelled)     

        rg_mask |= (small_mask == 1)

    rg_mask = skim.morphology.binary_closing(rg_mask, skim.morphology.disk(5))
    rg_mask = skim.morphology.remove_small_holes(rg_mask, area_threshold=1000)
    rg_mask = skim.morphology.remove_small_objects(rg_mask, min_size=50)
    return rg_mask

def watershed(mask):
    distance = scipy.ndimage.distance_transform_edt(mask)
    coordinates = skim.feature.peak_local_max(distance, labels=mask, footprint=skim.morphology.disk(20))

    local_maxi = np.zeros_like(distance, dtype=bool)
    local_maxi[tuple(coordinates.T)] = True

    markers = scipy.ndimage.label(local_maxi)[0]

    labels = skim.segmentation.watershed(-distance, markers, mask=mask)
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
