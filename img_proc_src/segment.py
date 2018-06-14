import cv2
import numpy as np
import peakutils
import sys
import os

def compare(img_above,img_below):
    diff_thresh = 2*255
    diff = img_above - img_below
    
    if diff > diff_thresh:
#         print('img_above:',img_above,type(img_above))
#         print('img_below:',img_below,type(img_below))
#         print('diff = ',diff)
        return -1
    
    elif diff < -diff_thresh:
#         print('img_above:',img_above,type(img_above))
#         print('img_below:',img_below,type(img_below))
#         print('diff = ',diff)
        return 1
    
    else:
        return 0


def run(image_name):

    #Load the image.....
    
    binary_img_path = os.path.join("Binarized_Input_Images/",image_name)
    
    color_img = cv2.imread(binary_img_path)
    
    gray_img = cv2.cvtColor(color_img,cv2.COLOR_RGB2GRAY)
    
    mfiltered = cv2.medianBlur(gray_img,5)
    
    gauss = cv2.GaussianBlur(mfiltered,(51,3),0)
    
    #cv2.imwrite("mfiltered.jpg",mfiltered)
    
    #cv2.imwrite("gaussian_blur.jpg",gauss)

    trace = np.array(color_img,copy=True)

    im_width = gauss.shape[1]

    left_half = gauss[:,25:int(0.5*im_width)+1]
    right_half = gauss[:,int(0.5*im_width):im_width-25]

    right_projection = right_half.sum(axis=1).astype(np.float64)
    left_projection = left_half.sum(axis=1).astype(np.float64)

    min_dist = 80

    right_indices = peakutils.indexes(right_projection, thres=0.03/max(right_projection), min_dist=min_dist)
    left_indices = peakutils.indexes(left_projection, thres=0.03/max(left_projection), min_dist=min_dist)

    n = 6; nby2 = 3; D = 12
    im_height,im_width = gauss.shape

    #Left to Right!
    #for i in range(25,im_height-25,inter_line_distance):

    lines = []

    for i in left_indices:
        line = []

        for j in range(25,im_width):

            #row, so if it starts going up, then this is the limit...
            if i == 12:
                continue

            c_up = i-D; c_down = i+D
            img_above = int(gauss[c_up-nby2:c_up+nby2,j-2*nby2:j+2*nby2].sum())
            img_current = int(gauss[i-nby2:i+nby2,j-nby2:j+nby2].sum())
            img_below = int(gauss[c_down-nby2:c_down+nby2,j-2*nby2:j+2*nby2].sum())

            #If region is a black region
            if img_current < 250*12:
                line.append(i)
    #             color_img[i,j] = (0,255,0)
                continue

            move = compare(img_above,img_below)

            #Inorder to increase line thickness to 3
            trace[i-1,j] = (255,0,0)
            trace[i,j] = (255,0,0)
            trace[i+1,j] = (255,0,0)

            line.append(i)

            i+=move

        lines.append(line)

    #lines
    height,width,_ = trace.shape
    line_image = np.zeros((height,width))
    line_image.fill(255)

    #print(line_image.shape)
    #print(trace.shape)

    for line in lines:
        #Classic Error...
        col = 25
        #print(len(line),file=sys.stderr)
        for i in line:
            #print(point)
            row = i
            #print(row,col)
            line_image[row,col] = 0
            col += 1

    #cv2.imwrite("lines.jpg",line_image)

    #------------------------------------------------------------------------------------

    #Right to Left!
    # for i in range(25,im_height-25,inter_line_distance):
    # for i in right_indices:
    #     for j in range(im_width-25,int(0.5*im_width),-1):
    #         if i == 12:
    #             continue
    #         c_up = i-D; c_down = i+D
    #         img_above = int(gauss[c_up-nby2:c_up+nby2,j-2*nby2:j+2*nby2].sum())
    #         img_current = int(gauss[i-nby2:i+nby2,j-nby2:j+nby2].sum())
    #         img_below = int(gauss[c_down-nby2:c_down+nby2,j-2*nby2:j+2*nby2].sum())

    #         #If region is a black region
    #         if img_current < 250*12:
    # #             color_img[i,j] = (0,255,0)
    #             continue

    #         move = compare(img_above,img_below)
    #         trace[i-1,j] = (0,0,255)
    #         trace[i,j] = (0,0,255)
    #         trace[i+1,j] = (0,0,255)

    #         i+=move

        #     if move==1:
        #         if count%5==0:
        #             cv2.rectangle(color_img,(j-nby2,c_up-nby2),(j+nby2,c_up+nby2),(255,0,0),thickness=1)
        #             cv2.rectangle(color_img,(j-nby2,c_down-nby2),(j+nby2,c_down+nby2),(255,0,0),thickness=1)        
        #         count+=1
        #         print('Going Down at :',(i,j))

        #     elif move==-1:
        #         if count%5 == 0:
        #             cv2.rectangle(color_img,(j-nby2,c_up-nby2),(j+nby2,c_up+nby2),(255,255,0),thickness=1)
        #             cv2.rectangle(color_img,(j-nby2,c_down-nby2),(j+nby2,c_down+nby2),(255,255,0),thickness=1)
        #         count+=1
        #         print('Going Up at :',(i,j))

    #--------------------------------------------------------------------------------------------

    #cv2.imwrite('trace.jpg',trace)


    #Extract....
    #take the 1st two lines

    #loop for picking 2 lines at a time
    
    #Add topmost and bottom-most line
    #print("height = ",height,file=sys.stderr)
    
    if len(lines)!=0:
        lines.insert(0, [0]*len(lines[-1]))
        lines.append([height-1]*len(lines[-1]))
    
    #new_list = [None]*(len(lines)+1)
    
    #new_list = [lines[i+1] for i in range(len(lines))] 
    
    #new_lines = [0]*len(lines[0]) + lines + 
    
    #lines = new_lines
    
    #print("no of lines:",len(lines),file=sys.stderr)
    #print("lines[0],lines[1]:",len(lines[0]),len(lines[1]),file=sys.stderr)
    
    for i in range(1,len(lines)):
        
        #i-1 --> 1st line
        #i --> 2nd line
        #highest point of i-1
        upper_bound = min(lines[i-1])
        lower_bound = max(lines[i])


        line_height = lower_bound - upper_bound + 1

        #content = .sum()

        new_line = np.zeros((line_height,width))
        new_line.fill(255)
        
        #column number and lines[i-1][j] --> row no.
        for j in range(len(lines[i-1])):
            #lines[i-1][j] --> row no. of 1st line
            #lines[i][j] --> row no. of 2nd line
            #column --> starting row, ending row, starting width = ending width
            #something to track the column no.  
            #j keeps track of the column.
            #pick the jth column from row lines[i-1][j] to line[i][j]
            img_cross_section = mfiltered[lines[i-1][j]:lines[i][j],j]
            #img_cross_section = np.expand_dims(img_cross_section,axis = -1)
            #print("img_cross_section:",img_cross_section.shape)
            #upperbound --> 0 
            #lowerbound --> lower_bound - upper
            #lower_bound- end
            #start - upper_bound
            #new_line ka column ka ek part -> this thing...
            #new_line[r1:r2,j]
            r1 = lines[i-1][j] - upper_bound
            r2 = lines[i][j] - upper_bound

            #print("r1,r2 = ",r1,r2)
            #print("new_line:",new_line[ r1:r2 , j])
            new_line[ r1 :r2 , j] = img_cross_section

        new_line = new_line.astype(np.uint8)
        inv_img = ~new_line
        #print(inv_img)
        img_sum = np.sum(inv_img)
 
        #print("Line_{0} = {1}".format(i,img_sum),file=sys.stderr)
 
        num_digits = len(str(img_sum))

        if num_digits < 5:
            continue
        
        #print(os.getcwd())
        
        line_path = os.path.join("Segmented_Images/Raw_Input/line_"+str(i).zfill(2)+".jpg")
        
        #print("Saving file at ",os.path.join(os.getcwd(),line_path),file=sys.stderr)
        
        cv2.imwrite(line_path,new_line)
