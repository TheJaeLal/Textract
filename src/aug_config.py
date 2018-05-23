import numpy as np

myfilter = np.ones((3,3,1),dtype=np.float32)

img_folder_path = '../data/train'
out_folder_path = '../Augmented'

max_rotation = 3

zoom_width,zoom_height = [1.2,1.2] 

max_zoom = [zoom_width,zoom_height]

baseline_upper_bound = 70
baseline_lower_bound = 95
baseline_width = 2
