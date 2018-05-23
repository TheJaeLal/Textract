import numpy as np
import cv2
import os
import math
from img_config import input_path,output_path,max_width,max_height,scaled_width,scaled_height,thresh_kernel_size 

# import matplotlib.pyplot as plt
from skimage.filters import (median,threshold_sauvola)
from skimage.io import imread,imsave

images = [image for image in os.listdir(input_path) if image.lower().endswith('.jpg') or image.lower().endswith('.png')]

for image_name in images: 

    image = imread(os.path.join(input_path,image_name),as_grey=True)

    image = median(image)
    thresh = threshold_sauvola(image, window_size=thresh_kernel_size,k=0.15)
    binary = image > thresh
    
    #Convert binary image from bool to int
    binary = binary.astype(int)
    binary = binary*255
  
    image_height, image_width = image.shape
    
    padding_vertical = max_height - image_height
    padding_top = int(padding_vertical/2)

    new_image = np.zeros((max_height,max_width))
    new_image.fill(255)
    
    new_image[padding_top:padding_top+image_height,0:image_width] = binary

    resize_image = cv2.resize(new_image,(scaled_width,scaled_height,),interpolation = cv2.INTER_AREA)

    #Scaled original image width
#     scaled_image_width = math.ceil(image_width/3)
    
#     #Save image_name, width to file
#     f = open('scaled_widths.txt','a')
#     f.write("{0},{1}\n".format(image_name,scaled_image_width))
#     f.close()
    
#     resize_image.shape

#   imsave(os.path.join(output_path,'binarize_'+image_name),binary)
    cv2.imwrite(os.path.join(output_path,image_name),resize_image)