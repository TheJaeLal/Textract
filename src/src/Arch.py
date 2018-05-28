from train_config import vocab_size

#Format: 'conv':[filter_size,strides,num_filters], 'pool':[strides]

#Model Params
CNN = [

        {'conv':[3,1,64], 'activate':'leaky_relu', 'pool':(3,2),'padding':'SAME','batch_norm':False},
        {'conv':[3,1,128], 'activate':'leaky_relu', 'pool':(3,2),'padding':'SAME','batch_norm':False},
        {'conv':[3,1,256], 'activate':'leaky_relu', 'pool':None,'padding':'SAME','batch_norm':False},
        {'conv':[3,1,256], 'activate':'leaky_relu', 'pool':(3,1),'padding':'SAME','batch_norm':False},
        {'conv':[3,1,512], 'activate':'leaky_relu', 'pool':None,'padding':'SAME','batch_norm':True},
        {'conv':[3,1,512], 'activate':'leaky_relu', 'pool':(3,1),'padding':'SAME','batch_norm':True},
        {'conv':[2,1,512], 'activate':'leaky_relu', 'pool':None,'padding':'VALID','batch_norm':False},

    
#         {'conv':[3,2,64], 'activate':'leaky_relu', 'pool':None},
#         {'conv':[3,2,128], 'activate':'leaky_relu', 'pool':None},
#         {'conv':[3,1,200], 'activate':'leaky_relu' ,'pool':3}
]

BRNN = {
        'layers':[256,256]
}

#FC Layer between CNN and BRNN
Interim_FC = [
    
#         {'units':BRNN['layers'][0],'activate':CNN[-1]['activate']}
]

FC = [
        {'units':vocab_size, 'activate': None}
]