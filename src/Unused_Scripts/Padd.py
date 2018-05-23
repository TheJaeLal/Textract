import cv2
import shelve
import os
import numpy as np

#Cuz inside 'code' directory
mount_point = "../"

with shelve.open('shelved_data/IAM_Data','c') as shelf:
	#image_labels = shelf['image_labels']
	#threshold_dict = shelf['image_thresholds']
	#chars = shelf['chars']
	image_paths = shelf['list_of_images']

max_height = 345
max_width = 2295

# heights = []
# widths = []

padded_prefix = "Padded_"

for image_path in image_paths:

	#binary_path = binarized_prefix+image_path
	
	new_path = padded_prefix + image_path

	directory = os.path.dirname(new_path)
	
	#print(directory)
	
	#Does directory exist
	if not os.path.exists(directory):
		os.makedirs(directory)

	#Does image_exits
	if os.path.exists(new_path):
		print("Image already exists....")
		continue

# 	elif image_path in ["Words/r06/r06-022/r06-022-03-05.png","Words/a01/a01-117/a01-117-05-02.png","Words/r02/r02-060/r02-060-08-05.png"
# ] :
# 		print("Found_culprit")
# 		continue

	print(image_path)
	#print(binary_path)
	print(new_path)	

	image = cv2.imread(image_path)
	gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	image_height,image_width = gray_image.shape

	#Horizontal Padding..
	padding_left = 20
	padding_right = max_width - image_width

	#Vertical Padding..
	padding_v = max_height - image_height
	padding_top = int(padding_v/2)
	padding_bottom = padding_v - padding_top

	#Add Padding
	new_image = np.zeros((max_height,max_width))
	new_image.fill(255)
	new_image[padding_top:padding_top+image_height,padding_left:padding_left+image_width] = gray_image

	#cv2.imshow(new_image)

	cv2.imwrite(new_path,new_image)