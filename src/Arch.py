from config import vocab_size


#Cuz the file is inside 'code' directory
mount_point = "../"

with shelve.open(mount_point+'IAM_Data') as shelf:
    vocabulary = shelf['chars']
    # list_of_images = shelf['list_of_images']
    # image_labels = shelf['image_labels']
    
vocab_size = len(vocabulary)

#Format: 'conv':[filter_size,strides,num_filters], 'pool':[strides]

#Model Params
CNN = [
        {'conv':[3,1,25], 'activate':'relu', 'pool':2},
        {'conv':[3,1,50], 'activate':'relu', 'pool':2},
        {'conv':[3,1,100], 'activate':'relu', 'pool':2},
        {'conv':[3,1,200], 'activate':'relu', 'pool':3},
        {'conv':[3,1,400], 'activate':'relu' ,'pool':None},
]

BRNN = {
        'layers':5,
        'hidden_units':256,
}

FC = [
        {'units':2*BRNN['hidden_units'],'activate':None},
        {'units':2*vocab_size,'activate':None},
        {'units':vocab_size,'activate':None},
]