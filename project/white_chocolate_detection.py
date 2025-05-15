## YOUR CODE
from matplotlib.widgets import RangeSlider, Button
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import skimage as sk
from skimage.color import rgb2gray, rgb2hsv
import os
from typing import Callable
from datetime import datetime

image_path = '../data/chocolate-recognition-classic/references/Comtesse.JPG'
image_path2 = '../data/chocolate-recognition-classic/references/Jelly_White.JPG'

img = np.array(Image.open(image_path2))

def extract_rgb_channels(img):
    """
    Extract RGB channels from the input image.

    Args
    ----
    img: np.ndarray (M, N, C)
        Input image of shape MxN and C channels.
    
    Return
    ------
    data_red: np.ndarray (M, N)
        Red channel of input image
    data_green: np.ndarray (M, N)
        Green channel of input image
    data_blue: np.ndarray (M, N)
        Blue channel of input image
    """

    # Get the shape of the input image
    M, N, _ = np.shape(img)

    # Define default values for RGB channels
    data_red = np.zeros((M, N))
    data_green = np.zeros((M, N))
    data_blue = np.zeros((M, N))

    data_red = img[:, :, 0]
    data_green = img[:, :, 1]
    data_blue = img[:, :, 2]

    
    return data_red, data_green, data_blue

def extract_hsv_channels(img):
    """
    Extract HSV channels from the input image.

    Args
    ----
    img: np.ndarray (M, N, C)
        Input image of shape MxN and C channels.
    
    Return
    ------
    data_h: np.ndarray (M, N)
        Hue channel of input image
    data_s: np.ndarray (M, N)
        Saturation channel of input image
    data_v: np.ndarray (M, N)
        Value channel of input image
    """

    # Get the shape of the input image
    M, N, C = np.shape(img)

    # Define default values for HSV channels
    data_h = np.zeros((M, N))
    data_s = np.zeros((M, N))
    data_v = np.zeros((M, N))

    hsv_img = rgb2hsv(img)

    data_h = hsv_img[:,:,0]
    data_s = hsv_img[:,:,1]
    data_v = hsv_img[:,:,2]
    
    return data_h, data_s, data_v

M, N, C = np.shape(img)
mask_blood = np.zeros((M, N))
mask_mucin = np.zeros((M, N))

# ------------------
data_red, data_green, data_blue = extract_rgb_channels(img=img)
data_h, data_s, data_v = extract_hsv_channels(img=img)
hsvimg = rgb2hsv(img)
white = (data_h > 0.12) & (data_s > 0.02) & (data_v > 0.63)
mask = (data_h > 0.25) | (data_s > 0.4) | (data_v > 0.9)

white[mask] = 0

mask_mucin = (data_red > 150) & (data_red < 230) & (data_green > 150) & (data_green < 230) & (data_blue > 150) & (data_blue < 230)

#mask_blood = remove_objects(mask_blood, 600)
#mask_blood = remove_holes(mask_blood, 200)

#mask_mucin = remove_objects(mask_mucin, 750)
#mask_mucin = remove_holes(mask_mucin, 300)

# Load image
path_he2 = '../data/chocolate-recognition-classic/references/Comtesse.JPG'

# Check if folder and image exist
assert os.path.exists(path_he2), "Image not found, please check directory structure"
img_he2 = np.array(Image.open(path_he2))

# Display image
plt.figure(figsize=(14, 7))
#plt.imshow(img_he2)
plt.axis('off')
plt.tight_layout()
plt.imshow(white)
plt.show()
#
#img = show_exo2_figure()
 