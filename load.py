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
        
        train_array = {}; valid_array = {}; test_array = {}; zombie_array = {}
        train_data=[]; test_data=[]; valid_data=[]; zombie_imgs = []
        train_text=""; valid_text=""; test_text=""; zombie_text=""
        train_prefix = 'data/Images/train/'
        valid_prefix = 'data/Images/valid/'
        test_prefix = 'data/Images/test/'
        zombie_prefix = 'data/Images/zombie/'

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
                    
                    img_path = file_location_subdir + "/" + list_sub_dir[j] + "/" + img_file
                    
                    #Read the image
                    img = imread(img_path,as_grey=True)
                    # img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
                    
                    #Binarize the image with provided threshold
                    img = ((img > int(thresh_line)).astype(np.uint8))*255

                    #Store the corresponding labels in a dictionary
                    dict_data[img_file] = line_data

                    #print(img_file)

                    if img_file[:-4] in train_imgs:
                        train_data.append(img_file)
                        train_array[img_file] = img
                        train_text += line_data
                        plt.imsave(train_prefix+img_file,img,cmap='gray',format='jpg')
                        # cv2.imwrite(train_prefix+img_file,img)


                    elif img_file[:-4] in valid_imgs:
                        valid_data.append(img_file)
                        valid_array[img_file] = img
                        valid_text += line_data
                        plt.imsave(valid_prefix+img_file,img,cmap='gray',format='jpg')
                        # cv2.imwrite(valid_prefix+img_file,img)
                    
                    elif img_file[:-4] in test_imgs:
                        test_data.append(img_file)
                        test_array[img_file] = img
                        test_text += line_data
                        plt.imsave(test_prefix+img_file,img,cmap='gray',format='jpg')
                        # cv2.imwrite(test_prefix+img_file,img)

                    else :
                        zombie_imgs.append(img_file)
                        zombie_array[img_file] = img
                        zombie_text += line_data
                        plt.imsave(zombie_prefix+img_file,img,cmap='gray',format='jpg')
                        # cv2.imwrite(zombie_prefix+img_file,img)

               #      #Special char flag if special char in line_data 
               #      special_char = False
                    
               #  #    for char in to_remove:
               #  #        if char in line_data:
               #  #            special_char = True

               #      if not special_char:
               #          total_text += line_data
               #          # list_data.append(img_path)

               # #         for char in to_increase:
               # #             if char in line_data:
               # #                 for _ in range(10):
               #          # list_data.append(img_path)
               #          # total_text += line_data

                    count = count + 1
                    
        train_chars = set(train_text)
        valid_chars = set(valid_text)
        test_chars = set(test_text)
        zombie_chars = set(zombie_text)
             
        chars = [train_chars,valid_chars,test_chars,zombie_chars]
        images = [train_data,valid_data,test_data,zombie_imgs]
        arrays = [train_array,valid_array,test_array,zombie_array]
        return dict_data,chars,images,arrays


    
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
