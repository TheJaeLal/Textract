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

alpha = 1e-4
