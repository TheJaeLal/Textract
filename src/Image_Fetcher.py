import os
import cv2

def fetch(input_dir):
	"""Returns valid images (jpg or png),
		from the specified directory 
		as a list of numpy arrays
	"""
	image_list = os.lisdir(input_dir)

	#Vaidate Images
	image_list = [image_name for image_name in os.listdir(image_list) if image_name.lower().endswith('.jpg') or image_name.lower().endswith('.png')]

	image_list.sort()

	images = [ cv2.imread(os.path.join(input_dir,image_name),cv2.IMREAD_GRAYSCALE) for image_name in image_list]

	return images
