#!/usr/bin/env python
# coding: utf-8

# In[13]:


from keras import backend as K
from warmup import WarmUpCosineDecayScheduler
import cv2
from keras.models import load_model
import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

from statistics import mode
import time

start_time = time.time()

#############################
#root_path of model weights##
#############################
root_path = r'C:\Users\James\Documents\The project Miaowei Wang has done\Kaggle_NCFM-master\saved_models'
weights_path = os.path.join(root_path, 'weights2.h5')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Map probabilities to class names
label_names = ['Calling', 'Normal', 'Smoking', 'Smoking & Calling']

#name of source video
cap = cv2.VideoCapture('Taken_Phone_Speech.mp4')
box_size = 309
width = int(cap.get(3))

#Load the model weights
print('Loading model and weights from training process ...')
InceptionV3_model = load_model(weights_path)

#######################
#path of output video##
#######################
#place and name the output file
FILE_OUTPUT = 'C:/Users/James/Documents/The project Miaowei Wang has done/Kaggle_NCFM-master/'+str(time.time())+'_outvid.avi'

# Checks for and deletes the output file if it exists
if os.path.isfile(FILE_OUTPUT):
    os.remove(FILE_OUTPUT)


# Default resolutions of the video frames are obtained
# Convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

#define video output format
out = cv2.VideoWriter(FILE_OUTPUT, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                      30, (frame_width, frame_height))
 
smooth = 15 #number of frames to average the index and probability results by
framecount=0 #counts the number of frames processed in order to ensure the minimum for a prediction are used
type_framcount= [0]*4 #'Calling', 'Normal', 'Smoking', 'Smoking & Calling'

movArr = [0]*smooth #initialize the moving average array for proability 
movMode = [0]*smooth #initialize the moving mode array
i = 0 #initialize the counter for the average aand mode arrays

############################################
## Loop that runs through all video frames##
############################################

while True:
    
    ret, frame = cap.read()
    if not ret:
        break
    #invert the frame    
    frame = cv2.flip(frame, 1)
    
    #open a window to display the frame
    cv2.namedWindow("CVDancer", cv2.WINDOW_NORMAL)
    
    #define the image size the model accepts 
    dsize = (299, 299)

    # resize image to this size
    roi = cv2.resize(frame, dsize)
    
    # Normalize the image and convert float64 array.
    roi = np.array([roi]).astype('float64') / 255.0
 
    # Get model's prediction.
    pred = InceptionV3_model.predict_generator(roi)
    
    # Get the index of the target class.
    target_index = np.argmax(pred[0])
    #increase the framecount of our frame counter
    type_framcount[target_index] +=1

    # Get the probability of the target class
    prob = np.max(pred[0])
    
    movArr[i]= prob #set index of moving average to current frame probability
    movMode[i]= target_index # set index of moving mode to current frame index
    i+=1 #iterate moving index 
    probavg = np.average(movArr) #calculate moving average
    most_common = mode(movMode) #calculate moving mode
    frame = cv2.flip(frame, 1)
    
    if framecount > smooth: #if minimum number of frames have filled index
        # Show results on frame
        cv2.putText(frame, "Prediction: {} {:.2f}%".format(label_names[most_common], probavg*100 ),
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
    #reset our floating counter if at the end of bounds
    if i > smooth-2:
        i = 0
    
    
    if ret == True:
                # Saves for video
                out.write(frame)
                framecount +=1

                # Display the resulting frame
                #cv2.imshow('Charving Detection', frame)
    cv2.imshow("CVDancer", frame)
   
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
        
#################################        
#print data about run
print("--- %s seconds to run---" % (time.time() - start_time))
print("--- %s Frames---" % (framecount))
#'Calling', 'Normal', 'Smoking', 'Smoking & Calling'
print("--- %s percent Frames Calling---" % (100* type_framcount[0]/framecount))
print("--- %s percent Frames Normal---" % (100* type_framcount[1]/framecount))
print("--- %s percent Frames Smoking---" % (100* type_framcount[2]/framecount))
print("--- %s percent Frames Smoking & Calling---" % (100* type_framcount[3]/framecount))


#close all windows and frames when done
out.release()
cap.release()
cv2.destroyAllWindows()


# In[ ]:




