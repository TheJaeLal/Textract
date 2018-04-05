import numpy as np
import os
import xml.etree.ElementTree as ET
import pdb
from collections import Counter
from skimage.io import imread
import matplotlib.pyplot as plt
'''
The class is used to load data
'''

def save_img(img,path):
    #img_dir = os.path.dirname(path)
    #if not os.path.exists(img_dir):
    #   os.makedirs(img_dir)
    print('new_path = {}\n'.format(path))
    img = img.reshape(img.shape[:2])
    plt.imsave(path,img,cmap='gray',format='png')
    return

class loadData:
    def __init__(self):
        pass

    def load(self, file_location, file_xml_location):
        '''
        The method is used to load data as line images(i.e image is an entire line)
        '''
        # to_remove = ['+','/','&','*','#']
        to_remove = []

        # to_increase = ['Z','X','Q','7']
        to_increase = []

        with open('train_list','r') as train:
            train_imgs = set(train.read().splitlines())
            print('Number of training images = ',len(train_imgs))

        with open('test_list','r') as test:
            test_imgs = set(test.read().splitlines())
            print('Number of testing images = ',len(test_imgs))


        with open('valid_list','r') as valid:
            valid_imgs = set(valid.read().splitlines())
            print('Number of validation images = ',len(valid_imgs))

        list_xml= os.listdir(file_xml_location)
        list_dir = os.listdir(file_location)
        
        train_label = []; valid_label = []; test_label = []; zombie_label = []
        
        train_array = []; valid_array = []; test_array = []; zombie_array = []
        train_data=[]; test_data=[]; valid_data=[]; zombie_imgs = []
        train_text=""; valid_text=""; test_text=""; zombie_text=""
        train_prefix = 'data/train/'
        valid_prefix = 'data/valid/'
        test_prefix = 'data/test/'
        zombie_prefix = 'data/zombie/'

        dict_data = {};
        
        for i in range(len(list_dir)):
            file_location_subdir =  file_location + list_dir[i]
            list_sub_dir = os.listdir(file_location_subdir)
            for j in range (len(list_sub_dir)):
                list_files_subdir = os.listdir(file_location_subdir + "/" + list_sub_dir[j])
                xml_file = file_xml_location + list_sub_dir[j] + '.xml'
                tree = ET.parse(xml_file)
                root = tree.getroot()
                count= 0
                for tag in root.iter('line'):
                    thresh_line = tag.attrib['threshold']
                    line_data = ""
                    for sub_tag in tag.iter('word'):
                        line_data += sub_tag.attrib['text'] + " "
                    if count >= 10:
                        img_file = list_sub_dir[j] + '-' + str(count)+'.png'
                    else:
                        img_file = list_sub_dir[j] + '-'+'0'+str(count)+'.png'
                    
                    #Input Path
                    img_path = os.path.join(file_location,list_dir[i],list_sub_dir[j],img_file)
                    
                    #Save Path
                    # new_path = os.path.join(list_dir[i],list_sub_dir[j],img_file)
                    new_path = img_file

                    #Read the image
                    img = imread(img_path,as_grey=True)

                    #Binarize the image with provided threshold
                    img = ((img > int(thresh_line)).astype(np.uint8))*255
                   
                    #Reshaping so that channel is last
                    img = img.reshape(img.shape + (1,))

                    #Store the corresponding labels in a dictionary
                    dict_data[img_file] = line_data

                    #Ignore the .png or .jpg extension..
                    if img_file[:-4] in train_imgs:
                        new_path = os.path.join(train_prefix,new_path)
                        train_data.append(new_path)
                        train_array.append(img)
                        train_text += line_data
                        train_label.append(line_data)
                        # train_array[img_file] = img

                    elif img_file[:-4] in valid_imgs:
                        new_path = os.path.join(valid_prefix,new_path)
                        valid_data.append(new_path)
                        valid_array.append(img)
                        # valid_array[img_file] = img
                        valid_text += line_data
                        valid_label.append(line_data)
                    
                    elif img_file[:-4] in test_imgs:
                        new_path = os.path.join(test_prefix,new_path)
                        test_data.append(new_path)
                        test_array.append(img)
                        # test_array[img_file] = img
                        test_text += line_data
                        test_label.append(line_data)

                    else :
                        new_path = os.path.join(zombie_prefix,new_path)
                        zombie_imgs.append(new_path)
                        zombie_array.append(img)
                        # zombie_array[img_file] = img
                        zombie_text += line_data
                        zombie_label.append(line_data)

                    save_img(img,new_path)

                    count = count + 1
                    
        train_chars = set(train_text)
        valid_chars = set(valid_text)
        test_chars = set(test_text)
        zombie_chars = set(zombie_text)
        
        train_array = np.stack(train_array,axis=0)
        valid_array = np.stack(valid_array,axis=0)
        test_array = np.stack(test_array,axis=0)
        zombie_array = np.stack(zombie_array,axis=0)
        
        train_label = np.stack(train_label,axis=0)
        valid_label = np.stack(valid_label,axis=0)
        test_label = np.stack(test_label,axis=0)
        zombie_label = np.stack(zombie_label,axis=0)

        arrays = [train_array,valid_array,test_array,zombie_array]
        chars = [train_chars,valid_chars,test_chars,zombie_chars]
        images = [train_data,valid_data,test_data,zombie_imgs]
        labels = [train_label,valid_label,test_label,zombie_label]
        
        return dict_data,chars,images,arrays,labels


    
    def loadData_word(self, file_location, file_xml_location):

        '''
        The method is used to load data as word images rather than lines (i.e each image is a word)
        '''
        list_xml= os.listdir(file_xml_location)
        list_dir = os.listdir(file_location)
        list_data=[];
        total_text = ""
        dict_data = {}
        thresh_dict = {}
        for i in range(len(list_dir)):
            file_location_subdir =  file_location + list_dir[i]
            list_sub_dir = os.listdir(file_location_subdir)
            for j in range (len(list_sub_dir)):
                list_files_subdir = os.listdir(file_location_subdir + "/" + list_sub_dir[j])
                xml_file = file_xml_location + list_sub_dir[j] + '.xml'
                tree = ET.parse(xml_file)
                root = tree.getroot()
                count= 0
                count_line=0
                for tag in root.iter('line'):
                    thresh_line = tag.attrib['threshold']
                    line_data = ""
                    count_word=0

                    for sub_tag in tag.iter('word'):
                        
                        #Hackish solution to corrupt data
                        if sub_tag.attrib['id'] == "a01-117-05-02":
                            print("Found culprit")
                            continue
                                                        
                        word = sub_tag.attrib['text'] 
                        if count_line>=10:
                            str_count_line = '-' + str(count_line) 
                        else:
                            str_count_line = '-0' + str(count_line)
 
                        if count_word >= 10:
                            img_file = list_sub_dir[j] + str_count_line + '-' + str(count_word)+'.png'
                        else:
                            img_file = list_sub_dir[j] + str_count_line + '-'+'0'+str(count_word)+'.png'

                        img_path = file_location_subdir + "/" + list_sub_dir[j] + "/" + img_file
                        list_data.append(img_path)
                        dict_data[img_path] = word
                        thresh_dict[img_path] = thresh_line 
                        count_word +=1
                        total_text+= word
                    count_line+=1
                    count = count + 1
       
        # chars = set(total_text)      
        chars =  Counter(total_text)
        return dict_data,thresh_dict,chars,list_data
