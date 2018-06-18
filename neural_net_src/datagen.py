import shelve
import joblib
from train_config import mount_point,more_train_data,valid_set,batch_size,valid_batch_size,store_augmented
from Augment import train_datagen,valid_datagen
import os

#Global Variables..
num_train = None
num_valid = None

def _get_img_data(more_train_data):
    """Load Training and Validation Input Images (As Numpy Arrays)
    """
    
    train_array = joblib.load(os.path.join(mount_point,'data','train_arrays'))
    valid_array = joblib.load(os.path.join(mount_point,'data',valid_set+'_arrays'))

    #If additional data is needed...
    if more_train_data:
        new_array = joblib.load(os.path.join(mount_point,'data',more_train_data+'_arrays'))
        train_array = np.concatenate((train_array,new_array))
    
    return train_array,valid_array
    
def _get_label_data(more_train_data):
    """Load Training and Validation Labels/Targets..
    """

    with shelve.open(os.path.join(mount_point,'Metadata'),'c') as shelf:
        train_label = shelf['train_labels']
        valid_label = shelf[valid_set+'_labels']

    #   If more data needed...
        if more_train_data:
            new_label = shelf[more_train_data+'_labels']
            train_label = np.concatenate((train_label,new_label))

    return train_label,valid_label

def get_generators():
    """Returns training and Validation generators
    """
    global num_train,num_valid
    
    train_array,valid_array = _get_img_data(more_train_data)
    train_label,valid_label = _get_label_data(more_train_data)
    
    #Assign value to global variables..
    num_train = len(train_array)
    num_valid = len(valid_array)
    
    if store_augmented:
        train_generator = train_datagen.flow(train_array,train_label,
                        batch_size,save_to_dir=os.path.join(mount_point,'Augmented'), 
                        save_prefix='train')

    else:
        train_generator = train_datagen.flow(train_array,train_label,
                                            batch_size)

    valid_generator = valid_datagen.flow(valid_array,valid_label,valid_batch_size)

    return train_generator,valid_generator