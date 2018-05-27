import os
import cv2
import numpy as np

def fetch(input_dir):
	print("Input_dir = ",input_dir)
	"""Returns valid images (jpg or png),
		from the specified directory 
		as a list of numpy arrays
	"""

	#Vaidate Images
	image_list = [image_name for image_name in os.listdir(input_dir) if image_name.lower().endswith('.jpg') or image_name.lower().endswith('.png')]

	image_list.sort()

	images = [ cv2.imread(os.path.join(input_dir,image_name),cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0  for image_name in image_list]
	# for i,image_name in enumerate(image_list):
	# 	cv2.imwrite(image_name,images[i])

	return images
