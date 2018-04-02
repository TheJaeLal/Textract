from load import loadData
import shelve
#import cv2
#import joblib

images = '../HCR_Stuff/Lines/'
xml = '../HCR_Stuff/XML_Data/'
shelve_loc = './'

data = loadData()

dict_data,thresh_dict,train_chars,valid_chars,test_chars,zombie_chars,train_data,valid_data,test_data,zombie_imgs = data.load(images,xml)

# for el in pluses:
# 	print(el)
	
# #print(list_data)

# #to_remove = ["Words/r06/r06-022/r06-022-03-05.png","Words/a01/a01-117/a01-117-05-02.png","Words/r02/r02-060/r02-060-08-05.png"]


# # for image_path in to_remove:
# # 	list_data.remove(image_path)
# # 	dict_data.pop(image_path)
# # 	thresh_dict.pop(image_path)

# #Dictionary:

# #pad_prefix = "Padded_"

# print(len(list_data))

# #image_arrays = { image_path : cv2.imread(pad_prefix+image_path,cv2.IMREAD_GRAYSCALE) for image_path in list_data }


with shelve.open(shelve_loc+'IAM_Data','c',writeback=True) as shelf:
	shelf['image_labels'] = dict_data
	shelf['chars'] = len(train_chars)
	shelf['train_data'] = train_data
	shelf['valid_data'] = valid_data
	shelf['test_data'] = test_data
	shelf['zombie_data'] = zombie_imgs
	shelf['image_thresholds'] = thresh_dict

# #print("Done saving Image_labels,chars and list_of_images")

# #joblib.dump(image_arrays,"shelved_data/image_arrays",compress=True)

# #print(image_arrays)

# #print(list_data)
# # print(thresh_dict)
