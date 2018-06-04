import os
import shutil
from img_proc_src import segment, padd_n_scale, binarize
from neural_net_src import Recognize

def reset_raw_input():  
    
    os.chdir("/home/ubuntu/hcr-ann/")
    
    raw_input_path = 'Segmented_Images/Raw_Input/'
    
    if os.path.exists(raw_input_path):
        shutil.rmtree(raw_input_path)
    
    os.mkdir(raw_input_path)
    return
    
def reset_processed():  
    
    os.chdir("/home/ubuntu/hcr-ann/")
    
    raw_input_path = 'Segmented_Images/Padded_N_Scaled/'
    
    if os.path.exists(raw_input_path):
        shutil.rmtree(raw_input_path)
    
    os.mkdir(raw_input_path)
    return

def extract(image_path):   
    #Delete files inside Segmented_Images/Raw_Input and Padded_N_Scaled folder
    
    reset_raw_input()
    reset_processed()
    
    binarize.run(image_path)
    
    segment.run(image_path)
    
    padd_n_scale.run()
    
    model_input_path = 'Segmented_Images/Padded_N_Scaled/'
    
    output_text = Recognize.get_text(model_input_path)
    
    return output_text
