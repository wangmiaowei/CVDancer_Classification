#!/usr/bin/env python
# coding: utf-8

# In[3]:


from keras import backend as K
from warmup import WarmUpCosineDecayScheduler
import cv2
from keras.models import load_model
import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


import cv2,time

#############################
#root_path of model weights##
#############################
root_path = r'C:\Users\James\Documents\The project Miaowei Wang has done\Kaggle_NCFM-master\saved_models'
weights_path = os.path.join(root_path, 'weights2.h5')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# This list will be used to map probabilities to class names, Label names are in alphabetical order.
label_names = ['calling', 'normal', 'smoking', 'smoking_calling']

#############################
#image to predict          ##
#############################
cap = cv2.imread('drz_normal.jpg')
box_size = 309
height, width, channels = cap.shape

print('Loading model and weights from training process ...')
InceptionV3_model = load_model(weights_path)

#######################
#path of output image##
#######################
FILE_OUTPUT = 'C:/Users/James/Documents/The project Miaowei Wang has done/Kaggle_NCFM-master/'+str(time.time())+'_outpic.png'

# Checks and deletes the output file if it already exists
if os.path.isfile(FILE_OUTPUT):
    os.remove(FILE_OUTPUT)


# Default resolutions of the frame (system dependant). Convert the resolutions from float to integer.
frame_width = width
frame_height = height

frame = cap    
frame = cv2.flip(frame, 1)

cv2.namedWindow("CVDancer", cv2.WINDOW_NORMAL)
#################################################################
## This section performs classification of the entire image

#define the image size the model accepts 
dsize = (299, 299)

# resize image
roi = cv2.resize(frame, dsize)

# Normalize the image and convert to float64 array.
roi = np.array([roi]).astype('float64') / 255.0

# Get model's prediction.
pred = InceptionV3_model.predict_generator(roi)

# Get the index of the target class.
target_index = np.argmax(pred[0])
saved_target_index = target_index

# Get the probability of the target class
prob = np.max(pred[0])
saved_prob = prob

frame = cv2.flip(frame, 1)
# Show results
cv2.putText(frame, "Prediction: {} {:.2f}%".format(label_names[np.argmax(pred[0])], prob*100 ),
            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2, cv2.LINE_AA)

# Saves the image
cv2.imwrite(FILE_OUTPUT, frame)

# Display the resulting frame
#cv2.imshow('Charving Detection', frame)

cv2.imshow("CVDancer", frame)


###########################################
# This section perfroms basic localization
# the image is plit into a 4x4 grid
# each image tile is classified and if the classification
# matches the whole-image classification,the tile is highlighted
# the tiles are then stiched back together to re-form the image


img = cap
img2 = cap
height, width, channels = img.shape
# Number columns to split the image into
CROP_W_SIZE  = 4 
# Number of rows to split the image into
CROP_H_SIZE = 4 

xdata = [] #make list/array to store image tiles in


for ih in range(CROP_H_SIZE ):
    for iw in range(CROP_W_SIZE ):
        #calculate coordiates of image tiles
        x = width//CROP_W_SIZE * iw 
        y = height//CROP_H_SIZE * ih
        h = (height // CROP_H_SIZE)
        w = (width // CROP_W_SIZE )
        print(x,y,h,w)
        img = img[y:y+h, x:x+w]
        
        #calculate a tile's label and probability        
        dsize = (299, 299)
        # resize image
        roi = cv2.resize(img, dsize)
        # Normalize the image like we did in the preprocessing step, also convert float64 array.
        roi = np.array([roi]).astype('float64') / 255.0
        # Get model's prediction.
        pred = InceptionV3_model.predict_generator(roi)
        # Get the index of the target class.
        target_index = np.argmax(pred[0])
        # Get the probability of the target class
        prob = np.max(pred[0])
        
        #If the tile matches the overall image classification, and probability is above 30%, highlight the tile
        if target_index == saved_target_index and prob>0.3:
            cv2.rectangle(img,(0,0),(w,h),(0,255,0),3)
            cv2.putText(img, "{:.2f}%".format(prob*100 ),
            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            
        #else:
            #cv2.rectangle(img,(0,0),(w,h),(0,0,0),3)
            
        
        #store each image tile in an array
        xdata.append (img)

        #save cropped pice for checking tile functionality
        #cv2.imwrite("CROP/" + str(time.time()) +  ".png",img)
        
        img = img2
        
#reconstruct the image tiles into original image    
row1 = np.concatenate((xdata[0], xdata[1], xdata[2], xdata[3]), axis=1)
row2 = np.concatenate((xdata[4], xdata[5], xdata[6], xdata[7]), axis=1)
row3 = np.concatenate((xdata[8], xdata[9], xdata[10], xdata[11]), axis=1)
row4 = np.concatenate((xdata[12], xdata[13], xdata[14], xdata[15]), axis=1)

reconstruct = np.concatenate((row1, row2, row3, row4), axis=0)
cv2.putText(reconstruct, "prediction: {} {:.2f}%".format(label_names[saved_target_index], saved_prob*100 ),
            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2, cv2.LINE_AA)
#save the reconstructed image
cv2.imwrite(str(time.time()) + "_image_4by4_localization" +  ".png",reconstruct)
#clear the image tile list
del xdata[:]
############################################################
              

k = cv2.waitKey(1)
if k == ord('q'):
    
    img.release()
    img2.release()
    reconstruct.release()
    out.release()
    cap.release()
    cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:




