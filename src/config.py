#Cuz the file is inside 'code' directory
mount_point = "../"

with shelve.open(mount_point+'IAM_Data') as shelf:
    vocabulary = shelf['chars']
    
vocab_size = len(vocabulary)

#Model parameters
img_height = 104
img_width = 688
