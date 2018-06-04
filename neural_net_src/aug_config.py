#Size of the morphological filter (for erosion and dilation)
filter_size = (2,2)

#Relative Paths...
img_folder_path = '../data/train'
out_folder_path = '../Augmented'

#Zoom Range
zr = 0.05

#height-shift range
hsr = 0.05

#width-shift range
wsr = 0.07

#shear range
shr = 0.01

#Rotation range
ror = 2.5

#For introducing a blank baseline (making img similar to baseline-removed imgs)
baseline_upper_bound = 70
baseline_lower_bound = 95
baseline_width = 2