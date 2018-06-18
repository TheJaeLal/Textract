import shelve
import os
#import sys

#Cuz the file is inside 'neural_net_src' directory
mount_point = "/home/ubuntu/Textract/"

#print(os.getcwd())

#print(sys.path)

#data_path = mount_point + 'IAM_Metadata.dat'

# if os.path.exists(data_path):
#     print("file exists in path..")
# else:
#     print("file doesn't exist in path")

with shelve.open(os.path.join(mount_point,'Metadata')) as shelf:
    #print("Following are the contents of Metadata")
    #print(list(shelf.keys()))
    vocabulary = list(shelf['chars'])

vocabulary.sort()

print("Vocabulary:",vocabulary)

#Add Extra for the blank symbol
vocab_size = len(vocabulary)+1

#Model parameters
img_height = 750
img_width = 1000

#Dropout config...
dropout = { 'conv':[1,1,1,0.5,0.5,0.5,0.5],
           'lstm':0.5,
           'interim_fc':0.5,
           'fc':1
          }

#Toggle Data Augmentation
augment_data = False
#***** Do not remove augmentation, the code will break!!
store_augmented = False

#Use addition train_data ('zombie', test', or 'valid' images)
more_train_data = None

#Which set to use for Validation 'valid' or 'test'
valid_set = 'test'

#Regularization (weight decay)
decay = 0.0

#Optimizer Params
alpha = 0.0003

#Not in use yet..
momentum = 0.0

batch_size = 4
valid_batch_size = 4

#Training parameters
n_epochs = 1000
resume_epoch = 0
save_epoch = 1

#Tensorboard Summary (summary_epoch either None for no summary or an integer number say 'n'
#indicating summary is taken every 'n' epochs..
summary_epoch = None

#Not using as of now!
#CLR parameters
first_decay_steps = 800
t_mul = 2.0
m_mul = 1.0

#LSTM clipping
cell_clip = -1