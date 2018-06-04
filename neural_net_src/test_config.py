import shelve
import os

#Cuz the file is inside 'neural_net_src' directory
mount_point = "../"

os.chdir("/home/ubuntu/hcr-ann/neural_net_src")

with shelve.open(os.path.join(mount_point,'IAM_Metadata')) as shelf:
    vocabulary = list(shelf['chars'])
    
vocabulary.sort()

#Add Extra for the blank symbol
vocab_size = len(vocabulary)+1

#Model parameters
img_height = 114
img_width = 758
 
#Passed to the get_text() method in Recognize.py as a parameter
#input_dir = 'Test_Images/Processed'

test_batch_size = 64
infer_batch_size = 1

#Which model to restore?
model_dir = 'saved_models'
model_prefix = 'cnn_lstm_fc_'
resume_epoch = 140

#Set on which prediction is to be done valid/test
prediction_set = 'valid'
