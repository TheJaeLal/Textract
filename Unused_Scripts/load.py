import numpy as np
import os
import xml.etree.ElementTree as ET

from collections import Counter

def paragraph_labels(image_location, xml_location):

    image_names = [fname for fname in os.listdir(image_location) 
                    if fname.lower().endswith(".png") or fname.lower().endswith(".jpg")]

    image_names.sort()

    image_labels = []
    
    total_text = ""

    for image_name in image_names:
        base_name = image_name[:-4]

        xml_name = base_name+".xml"
        print(xml_name)

        tree = ET.parse(os.path.join(xml_location,xml_name))
        root = tree.getroot()
        count= 0         

        image_text = ""

        for tag in root.iter('line'):

            #thresh_line = tag.attrib['threshold']
            line_text = tag.attrib['text']
            
            total_text += line_text
                
            image_text += line_text + "\n"


        #Append to image_labels and remove the trailing whitespace characters
        image_labels.append(image_text.strip())
        

    # chars = set(total_text)
    chars =  Counter(total_text)

    return image_names,image_labels,chars