from neural_net_src.train_config import vocab_size

#Format: 'conv':[filter_size,strides,num_filters], 'pool':[pool_height,pool_width]

#Model Params


iterations = 32
#750 x 1000 --> 32 x 250
CNN = [

        {'conv':[3,1,64], 'activate':'leaky_relu', 'pool':(3,2),'padding':'SAME','batch_norm':False},
        {'conv':[3,1,128], 'activate':'leaky_relu', 'pool':(2,2),'padding':'SAME','batch_norm':False},
        {'conv':[3,1,256], 'activate':'leaky_relu', 'pool':None,'padding':'SAME','batch_norm':False},
        {'conv':[3,1,256], 'activate':'leaky_relu', 'pool':(2,1),'padding':'SAME','batch_norm':False},
        {'conv':[3,1,512], 'activate':'leaky_relu', 'pool':None,'padding':'SAME','batch_norm':True},
        {'conv':[3,1,512], 'activate':'leaky_relu', 'pool':(2,1),'padding':'SAME','batch_norm':True},
        {'conv':[3,1,512], 'activate':'leaky_relu', 'pool':None,'padding':'SAME','batch_norm':False},

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