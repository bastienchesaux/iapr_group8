import skimage as skim
import sklearn 
import cv2
import numpy as np
#import os # NO
import csv

import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import napari # NO
import _utils


if __name__ == "__main__":
    # table with name of chocolate
    chocolates = {
        0:  'Amandina',
        1:  'Arabia',
        2:  'Comtesse',
        3:  'Creme_brulee',
        4:  'Jelly_Black',
        5:  'Jelly_Milk',
        6:  'Jelly_White',
        7:  'Noblesse',
        8:  'Noir_authentique',
        9:  'Passion_au_lait',
        10: 'Stracciatella',
        11: 'Tentation_noir',
        12: 'Triangolo'
    }

    # directory paths
    reference_path = '../data/dataset_project_iapr2025/references'
    annotation_path = '../data/dataset_project_iapr2025/references_annotated'

    # full path and load images
    ref_choco = []
    ref_choco_annotated = []
    for i in range(len(chocolates)):
        img = np.array(Image.open(reference_path+'/'+chocolates[i]+'.jpg'))
        binned_ref = _utils.median_bin_rgb(img, block_size=(10, 10))
        ref_choco.append(binned_ref)

        img_annotated = np.array(Image.open(annotation_path+'/'+chocolates[i]+'_annotated.jpg'))
        binned_ref_annotated = skim.measure.block_reduce(img_annotated, block_size=(10, 10), func=np.median)
        N, M = binned_ref_annotated.shape
        for j in range(N):
            for k in range(M):
                if binned_ref_annotated[j][k] != 0:
                    binned_ref_annotated[j][k] = 1
                else:
                    binned_ref_annotated[j][k] = 0
        ref_choco_annotated.append(binned_ref_annotated)

    feat_ref = []
    names = ['Arabia', 'Jelly_Black', 'Jelly_Milk', 'Noblesse', 'Noir_authentique', 'Passion_au_lait', 'Stracciatella', 'Tentation_noir', 'Triangolo']
    
    for name in names:
        ref = np.array(Image.open(reference_path+'/'+name+'.jpg'))
        annot = np.array(Image.open(annotation_path+'/'+name+'_annotated.jpg'), dtype=bool)

        binned = _utils.median_bin_rgb(ref, block_size=(10, 10))
        annot = skim.measure.block_reduce(annot, block_size=(10,10), func=np.median)

        binned_bgr = cv2.cvtColor(binned, cv2.COLOR_RGB2BGR)
        binned_hsv = cv2.cvtColor(binned_bgr, cv2.COLOR_BGR2HSV)

        spx = skim.segmentation.slic(binned, n_segments=5000, compactness=.1, sigma=1, start_label=0)

        hsv_feat = _utils.hsv_spx_feat(binned_hsv, spx, np.median)
        h_ref = binned_hsv[annot>=1][:,0]/180
        s_ref = binned_hsv[annot>=1][:,1]/255
        v_ref = binned_hsv[annot>=1][:,2]/255

        hx_ref = np.cos(h_ref*2*np.pi)
        hy_ref = np.sin(h_ref*2*np.pi)
        
        feat_ref.append(np.array([hx_ref, hy_ref, s_ref, v_ref]).T)
    
    feat_ref = np.vstack(feat_ref)
    gmm_ref = sklearn.mixture.GaussianMixture(n_components=1, covariance_type='full', reg_covar=1e-3)
    gmm_ref.fit(feat_ref)

    with open('sample_submission.csv', mode='r') as file:
        reader = csv.reader(file)
        rows = list(reader)

    test_data_path = '../data/dataset_project_iapr2025/test'
    save_data_path = '../data/dataset_project_iapr2025/binary_masks'
    for row in rows[1:]:
        filename = '/L' + row[0] + '.jpg'
        filename2 = '/L' + row[0] + '_2' + '.jpg'
        full_path = test_data_path + filename

        img = np.array(Image.open(full_path))
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

        plt.figure(figsize=(9,6))
        plt.imshow(binned)
        plt.imshow(binary, alpha=0.7, cmap='Reds', interpolation='none')
        plt.axis('off')
        plt.savefig(save_data_path+'/'+filename)
        plt.close()

        rg_mask = _utils.region_growing(binned, binned_hsv, binary)

        labels = _utils.watershed(rg_mask)

        labels *= rg_mask

        plt.figure(figsize=(9,6))
        plt.imshow(binned)
        plt.imshow(labels, alpha=0.7, cmap='inferno')
        plt.axis('off')
        plt.savefig(save_data_path+'/'+filename2)
        plt.close()

        nb_chocolate = _utils.classifier(binned, labels, ref_choco, ref_choco_annotated)
        print(nb_chocolate)

        for i in range(len(ref_choco)):
            row[i+1] = str(int(nb_chocolate[i]))

        # Save the modified rows back to the CSV file
        with open('sample_submission.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(rows[0])  # Write the header
            writer.writerows(rows[1:])  # Write the modified rows


