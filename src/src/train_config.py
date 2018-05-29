import shelve

#Cuz the file is inside 'code' directory
mount_point = "../"

with shelve.open(mount_point+'IAM_Data') as shelf:
    vocabulary = list(shelf['chars'])
    
vocabulary.sort()

#Add Extra for the blank symbol
vocab_size = len(vocabulary)+1

#Model parameters
img_height = 114
img_width = 758

#Dropout config...
dropout = { 'conv':[1,1,1,0.5,0.5,0.5,0.5],
           'lstm':0.5,
           'interim_fc':0.5,
           'fc':1
          }

#Toggle Data Augmentation
augment_data = True
store_augmented = False

#Use addition train_data ('zombie', test', or 'valid' images)
more_train_data = None

#Which set to use for Validation 'valid' or 'test'
valid_set = 'valid'

#Regularization (weight decay)
decay = 0.0

#Optimizer Params
alpha = 0.0005

#Not in use yet..
momentum = 0.0

batch_size = 48
valid_batch_size = 48

#Training parameters
n_epochs = 1000
resume_epoch = 0
save_epoch = 1

#Tensorboard Summary
summary_epoch = 5

#Not using as of now!
#CLR parameters
first_decay_steps = 800
t_mul = 2.0
m_mul = 1.0

#LSTM clipping
cell_clip = -1