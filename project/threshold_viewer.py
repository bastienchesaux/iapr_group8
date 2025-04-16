from matplotlib.widgets import RangeSlider, Button
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import skimage as sk

image_path = '../data/dataset_project_iapr2025/train/L1000756.JPG'

img = np.array(Image.open(image_path))

use_hsv = False
revert = [False, False, False]

if use_hsv:
    hsv = sk.color.rgb2hsv(img)

mean = np.mean(img, axis=(0, 1))
low_thresh = 0.7*mean
high_thresh = 1.3*mean

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.4)

def threshold(img):
    mask = np.zeros_like(img, dtype=bool)
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



