from matplotlib.widgets import RangeSlider, Button
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import skimage as sk
import cv2

def median_bin_rgb(image, block_size=(2, 2)):
    channels = []
    for i in range(3):  # R, G, B
        channel = sk.measure.block_reduce(image[:, :, i], block_size=block_size, func=np.median)
        channels.append(channel)
    return np.stack(channels, axis=-1).astype(np.uint8)

image_path = '../data/dataset_project_iapr2025/train/L1000993.JPG'



img = np.array(Image.open(image_path))

binned = median_bin_rgb(img, block_size=(10, 10))
binned = sk.filters.gaussian(binned, sigma=.4, channel_axis=2, preserve_range=True).astype(np.uint8)


use_hsv = True
revert = [False, False, False]

if use_hsv:
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

mean = np.mean(img, axis=(0, 1))
low_thresh = [0, 0, 0]
high_thresh = [255, 255, 255]

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.4)

def threshold(img):
    mask = np.zeros_like(img, dtype=bool)
    if not use_hsv:
        if not revert[0]:
            mask = (img[..., 0] > low_thresh[0]) & (img[..., 0] < high_thresh[0])
        else:
            mask = (img[..., 0] < low_thresh[0]) | (img[..., 0] > high_thresh[0])

        if not revert[1]:
            mask &= (img[..., 1] > low_thresh[1]) & (img[..., 1] < high_thresh[1])
        else:
            mask &= (img[..., 1] < low_thresh[1]) | (img[..., 1] > high_thresh[1])
        if not revert[2]:
            mask &= (img[..., 2] > low_thresh[2]) & (img[..., 2] < high_thresh[2])
        else:
            mask &= (img[..., 2] < low_thresh[2]) | (img[..., 2] > high_thresh[2])
    else:
        if not revert[0]:
            mask = (hsv[..., 0] > low_thresh[0]) & (hsv[..., 0] < high_thresh[0])
        else:
            mask = (hsv[..., 0] < low_thresh[0]) | (hsv[..., 0] > high_thresh[0])

        if not revert[1]:
            mask &= (hsv[..., 1] > low_thresh[1]) & (hsv[..., 1] < high_thresh[1])
        else:
            mask &= (hsv[..., 1] < low_thresh[1]) | (hsv[..., 1] > high_thresh[1])
        if not revert[2]:
            mask &= (hsv[..., 2] > low_thresh[2]) & (hsv[..., 2] < high_thresh[2])
        else:
            mask &= (hsv[..., 2] < low_thresh[2]) | (hsv[..., 2] > high_thresh[2])

    overlay = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
    overlay[mask] = [0, 0, 255, 200]  # Red color for the overlay
    
    return overlay

ax.imshow(img)
ax.imshow(threshold(img))



slider_r_ax = plt.axes([0.2, 0.33, 0.6, 0.03])
slider_g_ax = plt.axes([0.2, 0.26, 0.6, 0.03])
slider_b_ax = plt.axes([0.2, 0.19, 0.6, 0.03])

slider_r = RangeSlider(slider_r_ax, 'Red', 0, 255, valinit=(low_thresh[0], high_thresh[0]))
slider_g = RangeSlider(slider_g_ax, 'Green', 0, 255, valinit=(low_thresh[0], high_thresh[0]))
slider_b = RangeSlider(slider_b_ax, 'Blue', 0, 255, valinit=(low_thresh[0], high_thresh[0]))

button1_ax = plt.axes([0.2, 0.1, 0.15, 0.05])
button2_ax = plt.axes([0.425, 0.1, 0.15, 0.05])
button3_ax = plt.axes([0.65, 0.1, 0.15, 0.05])

button1 = Button(button1_ax, 'revert r')
button2 = Button(button2_ax, 'revert g')
button3 = Button(button3_ax, 'revert b')

def update(val=None):
    low_thresh[0] = slider_r.val[0]
    high_thresh[0] = slider_r.val[1]
    low_thresh[1] = slider_g.val[0]
    high_thresh[1] = slider_g.val[1]
    low_thresh[2] = slider_b.val[0]
    high_thresh[2] = slider_b.val[1]

    ax.clear()
    ax.imshow(img)
    ax.imshow(threshold(img))
    fig.canvas.draw_idle()

def revert_r(event):
    revert[0] = not revert[0]
    update()

def revert_g(event):
    revert[1] = not revert[1]
    update()

def revert_b(event):
    revert[2] = not revert[2]
    update()
button1.on_clicked(revert_r)
button2.on_clicked(revert_g)
button3.on_clicked(revert_b)

slider_r.on_changed(update)
slider_g.on_changed(update) 
slider_b.on_changed(update)


plt.show()



