import skimage as skim
import sklearn 
import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt
from PIL import Image

import src._utils as _utils
import src.config as config


def compute_ref_choco():
    ref_choco = []
    for i in range(len(config.chocolates)):
        img = np.array(Image.open(config.reference_path+'/'+config.chocolates[i]+'.JPG'))
        binned_ref = _utils.median_bin_rgb(img, block_size=(10, 10))
        ref_choco.append(binned_ref)
    return ref_choco

def compute_ref_choco_annotated():
    ref_choco_annotated = []
    for i in range(len(config.chocolates)):
        img = np.array(Image.open(config.annotated_path+'/'+config.chocolates[i]+'_annotated.jpg'))
        binned_ref_annotated = skim.measure.block_reduce(img, block_size=(10, 10), func=np.median)
        N, M = binned_ref_annotated.shape
        for j in range(N):
            for k in range(M):
                if binned_ref_annotated[j][k] != 0:
                    binned_ref_annotated[j][k] = 1
                else:
                    binned_ref_annotated[j][k] = 0
        ref_choco_annotated.append(binned_ref_annotated)
    return ref_choco_annotated

def compute_gmm_ref():
    feat_ref = []
    for name in config.names:
        ref = np.array(Image.open(config.reference_path+'/'+name+'.jpg'))
        annot = np.array(Image.open(config.annotated_path+'/'+name+'_annotated.jpg'), dtype=bool)

        binned_ref = _utils.median_bin_rgb(ref, block_size=(10, 10))
        annot = skim.measure.block_reduce(annot, block_size=(10,10), func=np.median)

        binned_bgr_ref = cv2.cvtColor(binned_ref, cv2.COLOR_RGB2BGR)
        binned_hsv_ref = cv2.cvtColor(binned_bgr_ref, cv2.COLOR_BGR2HSV)

        spx = skim.segmentation.slic(binned_ref, n_segments=5000, compactness=.1, sigma=1, start_label=0)

        hsv_feat = _utils.hsv_spx_feat(binned_hsv_ref, spx, np.median)
        h_ref = binned_hsv_ref[annot>=1][:,0]/180
        s_ref = binned_hsv_ref[annot>=1][:,1]/255
        v_ref = binned_hsv_ref[annot>=1][:,2]/255

        hx_ref = np.cos(h_ref*2*np.pi)
        hy_ref = np.sin(h_ref*2*np.pi)
        
        feat_ref.append(np.array([hx_ref, hy_ref, s_ref, v_ref]).T)
    
    feat_ref = np.vstack(feat_ref)
    gmm_ref = sklearn.mixture.GaussianMixture(n_components=1, covariance_type='full', reg_covar=1e-3)
    gmm_ref.fit(feat_ref)
    return gmm_ref

def compute_nb_chocolate(row, ref_choco, ref_choco_annotated, gmm_ref):
        filename = '/L' + row[0] + '.JPG'
        full_path = config.test_data_path + filename

        img = np.array(Image.open(full_path))

        #chocolate-background segmentation
        binned = _utils.median_bin_rgb(img, block_size=(10, 10))

        binned_hsv = _utils.rgb_to_hsv(binned)

        spx = skim.segmentation.slic(binned, n_segments=10000, compactness=.05, sigma=0.1, start_label=0)

        hsv_feat = _utils.hsv_spx_feat(binned_hsv, spx, np.median)

        gmm = _utils.gmm_on_spx_fit(range(1,10), hsv_feat)
        labels = _utils.gmm_on_spx_predict(gmm, hsv_feat[spx], spx)

        kl_divergences = np.full(np.max(labels)+1, np.inf)
        for i in range(np.max(labels)+1):
            if len(binned_hsv[labels==i]) > 1000 and np.linalg.eigvals(gmm.covariances_[i]).max()<0.7:
                kl_divergences[i] = _utils.symmetric_kl(gmm.means_[i], gmm.covariances_[i], gmm_ref.means_[0], gmm_ref.covariances_[0])

        best_label = np.argmin(kl_divergences)
        merged = _utils.merge_labels(labels, best_label, 0.25)
        binary = merged==best_label


        #region growing & watershed
        rg_mask = _utils.region_growing(binned, binned_hsv, binary)

        labels = _utils.watershed(rg_mask)
        labels *= rg_mask

        #classifier
        nb_chocolate = _utils.classifier(binned, labels, ref_choco, ref_choco_annotated)
        return(nb_chocolate)

if __name__ == "__main__":

    ref_choco = compute_ref_choco()
    ref_choco_annotated = compute_ref_choco_annotated()

    gmm_ref = compute_gmm_ref()

    with open('sample_submission.csv', mode='r') as file:
        reader = csv.reader(file)
        rows = list(reader)

    for row in rows[1:]:
        nb_chocolate = compute_nb_chocolate(row, ref_choco, ref_choco_annotated, gmm_ref)

        for i in range(len(ref_choco)):
            row[i+1] = str(int(nb_chocolate[i]))

        with open('sample_submission.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(rows[0])
            writer.writerows(rows[1:])


