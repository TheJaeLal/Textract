import shelve

#Cuz the file is inside 'code' directory
mount_point = "../"

with shelve.open(mount_point+'IAM_Data') as shelf:
    vocabulary = list(shelf['chars'])
    
vocabulary.sort()
vocabulary.append('<Blank>')

vocab_size = len(vocabulary)

#Model parameters
img_height = 104
img_width = 688

alpha = 0.00256

batch_size = 32

#Training parameters
n_epochs = 10
resume_epoch = 0
save_epoch = 1