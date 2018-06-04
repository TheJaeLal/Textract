import os
import cv2
import numpy as np

def fetch(input_dir):
    #print("Fetching test_images for prediction from = ",input_dir)

    """Returns valid images (jpg or png),
        from the specified directory 
        as a list of numpy arrays
    """

    #Vaidate Images
    list_images = os.listdir(input_dir)
    list_images.sort()
    
    #print(list_images,file=sys.stderr)
    
    image_list = [image_name for image_name in list_images if image_name.lower().endswith('.jpg') or image_name.lower().endswith('.png')]

    image_list.sort()

    images = [ cv2.imread(os.path.join(input_dir,image_name),cv2.IMREAD_GRAYSCALE) for image_name in image_list]

    return images
