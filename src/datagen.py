import shelve
import joblib
from train_config import mount_point,use_more_data,valid_set,batch_size,valid_batch_size
from Augment import train_datagen,valid_datagen
import os

with shelve.open(mount_point+'IAM_Data','c') as shelf:
    train_label = shelf['train_label']
    
    valid_label = shelf[valid_set+'_label']

#   If more data needed use zombie data
    if use_more_data:
        zombie_label = shelf['zombie_label']

#Training and Validation Input Images (As Numpy Arrays)
train_array = joblib.load(os.path.join(mount_point,'data','train_array'))
valid_array = joblib.load(os.path.join(mount_point,'data',valid_set+'_array'))

#If additional data is needed...
if use_more_data:
    zombie_array = joblib.load(os.path.join(mount_point,'data','zombie_array'))

##*****Increase the training dataset...*****
##Add the zombie array to the train_array and do the same for labels...
if use_more_data:
    train_array = np.concatenate((train_array,zombie_array))
    train_label = np.concatenate((train_label,zombie_label))

##If you want to see augmented Images...
train_generator = train_datagen.flow(train_array,train_label,
                    batch_size,save_to_dir=os.path.join(mount_point,'Augmented'), 
                    save_prefix='train')

#train_generator = train_datagen.flow(train_array,train_label,batch_size)
valid_generator = valid_datagen.flow(valid_array,valid_label,valid_batch_size)

num_train = train_array.shape[0]
num_valid = valid_array.shape[0]