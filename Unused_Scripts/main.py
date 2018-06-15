import load
import shelve
#import cv2
#import joblib

# image_input_dir = '/home/james/Desktop/HCR_Stuff/cropped/'
# xml = '/home/james/Desktop/HCR_Stuff/XML_Data/'

# image_names,labels,chars= load.paragraph_labels(image_input_dir,xml)

# #print(chars)(
# print("  ")
# for i,label in enumerate(labels):
# 	print("*Label {0}* : {1}".format(i+1,label))
# 	print("----------------------------")

# with shelve.open("Metadata",'c',writeback=True) as shelf:
# 	shelf['images'] = image_names
# 	shelf['labels'] = labels
# 	shelf['chars'] = chars

#Verify stored Metadata...
# with shelve.open("Metadata",'c') as shelf:
# 	print(shelf['images'])

# 	print(shelf['labels'])

# 	print(shelf['chars'])


# #image_arrays = { image_path : cv2.imread(pad_prefix+image_path,cv2.IMREAD_GRAYSCALE) for image_path in list_data }


# with shelve.open(shelve_loc+'IAM_Data','c',writeback=True) as shelf:
# 	shelf['image_labels'] = dict_data
# 	shelf['chars'] = chars
# 	shelf['list_of_images'] = list_data
# 	shelf['image_thresholds'] = thresh_dict

# #print("Done saving Image_labels,chars and list_of_images")

# #joblib.dump(image_arrays,"shelved_data/image_arrays",compress=True)

# #print(image_arrays)

# #print(list_data)
# # print(thresh_dict)
