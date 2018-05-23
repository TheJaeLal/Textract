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

input_dir = 'test_images'

test_batch_size = 64
infer_batch_size = 1

resume_epoch = 37
model_dir = 'saved_models_ler0.07_leakyrelu'