import os
import os.path

import cv2
import numpy as np

import visualise_images


def read_all_images():
    images = []
    labels = []

    full_path = os.path.realpath(__file__)

    for x in range(0, 11):
        for y in range(0, 11):
            file_name = os.path.dirname(
                full_path
            ) + '/../live_demo/output/base-{:02d}-{:02d}.bmp'.format(x, y)
            if os.path.isfile(file_name):
                images.append(read_image(file_name))
                labels.append(x + 1)  #TODO: this is shitty if 1 image is skipped everything goes wrong.

    images = np.asarray(images)

    for i in range(0,30):
        #print(visualise_images.visualize_img(images[i]))
        #print(labels[i])
    return (images, labels)


def read_image(path):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    im[im == 255] = 0
    im[im == 29] = 1
    return im


read_all_images()
