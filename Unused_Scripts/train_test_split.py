import os
import shutil

home = '/home/ubuntu'
data_folder_path = os.path.join(home,'laia','egs','iam','data','imgs') 
img_folder_path = os.path.join(data_folder_path,'lines')

with open('train_list','r') as train:
    train_imgs = set(train.read().splitlines())
    print('Number of training images = ',len(train_imgs))

with open('test_list','r') as test:
    test_imgs = set(test.read().splitlines())
    print('Number of testing images = ',len(test_imgs))

with open('valid_list','r') as valid:
    valid_imgs = set(valid.read().splitlines())
    print('Number of validation images = ',len(valid_imgs))

        
train_array = []; valid_array = []; test_array = []; zombie_array = []
train_data=[]; test_data=[]; valid_data=[]; zombie_data = []

images = os.listdir(img_folder_path)

for image in images:

	if image[:-4] in train_imgs:
		train_data.append(image)
		dst = 'train'		

	elif image[:-4] in valid_imgs:
		valid_data.append(image)
		dst = 'valid' 

	elif image[:-4] in test_imgs:
		test_data.append(image)
		dst = 'test'

	else:
		zombie_data.append(image)
		dst = 'zombie'

	shutil.copy(os.path.join(img_folder_path,image),os.path.join(data_folder_path,dst))

print('Train data',len(train_data))
print('Valid data',len(valid_data))
print('Test data',len(test_data))
print('Zombie data',len(zombie_data))