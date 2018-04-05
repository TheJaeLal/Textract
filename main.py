from load import loadData
import shelve
import joblib

images = '../HCR_Stuff/Padded_Lines/'
xml = '../HCR_Stuff/XML_Data/'
shelve_loc = './'

data = loadData()

dict_data,chars,images,arrays,labels = data.load(images,xml)

train_array,valid_array,test_array,zombie_array = arrays
train_chars,valid_chars,test_chars,zombie_chars = chars
train_data,valid_data,test_data,zombie_imgs = images
train_label,valid_label,test_label,zombie_label = labels


# #to_remove = ["Words/r06/r06-022/r06-022-03-05.png","Words/a01/a01-117/a01-117-05-02.png","Words/r02/r02-060/r02-060-08-05.png"]


# # for image_path in to_remove:
# # 	list_data.remove(image_path)
# # 	dict_data.pop(image_path)
# # 	thresh_dict.pop(image_path)

# #Dictionary:

# #pad_prefix = "Padded_"

# print(len(list_data))

# image_arrays = { image_path : cv2.imread(pad_prefix+image_path,cv2.IMREAD_GRAYSCALE) for image_path in list_data }


with shelve.open(shelve_loc+'IAM_Data','c',writeback=True) as shelf:
    shelf['image_labels'] = dict_data
    shelf['chars'] = train_chars
    shelf['train_data'] = train_data
    shelf['valid_data'] = valid_data
    shelf['test_data'] = test_data
    shelf['zombie_data'] = zombie_imgs
    shelf['train_label'] = train_label
    shelf['valid_label'] = valid_label
    shelf['test_label'] = test_label
    shelf['zombie_label'] = zombie_label

joblib.dump(train_array,'train_arrays',compress=True)
joblib.dump(valid_array,'valid_arrays',compress=True)
joblib.dump(test_array,'test_arrays',compress=True)
joblib.dump(zombie_array,'zombie_arrays',compress=True)
