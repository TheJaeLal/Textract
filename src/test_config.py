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
            
input_dir = 'Test_Images/ImgenhOutput_Center'

test_batch_size = 64
infer_batch_size = 1

resume_epoch = 34
model_dir = 'saved_models'
model_prefix = 'cnn_lstm_fc_'

#Set on which prediction is to be done valid/test
prediction_set = 'valid'
