import os
import shutil
from Process_Images import segment, process, binarize
from src import Recognize

def reset_raw_input():  
    
    os.chdir("/home/ubuntu/hcr-ann/")
    
    raw_input_path = 'Test_Images/Raw_Input/'
    
    if os.path.exists(raw_input_path):
        shutil.rmtree(raw_input_path)
    
    os.mkdir(raw_input_path)
    return
    
def reset_processed():  
    
    os.chdir("/home/ubuntu/hcr-ann/")
    
    raw_input_path = 'Test_Images/Processed/'
    
    if os.path.exists(raw_input_path):
        shutil.rmtree(raw_input_path)
    
    os.mkdir(raw_input_path)
    return

def extract(image_path):   
    #Delete files inside Test_Images/Raw_Input folder
    reset_raw_input()
    reset_processed()
    
    binarize.run(image_path)
    
    segment.run(image_path)
    
    process.run()
    
    model_input_path = 'Test_Images/Processed'
    output_text = Recognize.get_text(model_input_path)
    
    return output_text
    
    
# if __name__ == '__main__':
#     extract()
