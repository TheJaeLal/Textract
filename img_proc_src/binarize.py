import numpy as np
import cv2

from skimage.filters import (median,threshold_sauvola)
from skimage.io import imread

from img_proc_src.img_config import thresh_kernel_size 


def run(image_path):

    image = imread(image_path,as_grey=True)

    image = median(image)
    thresh = threshold_sauvola(image, window_size=thresh_kernel_size,k=0.15)
    binary = image > thresh

    #Convert binary image from bool to int
    binary = binary.astype(int)
    binary = binary*255

    cv2.imwrite('Binarized_'+image_path,binary)
