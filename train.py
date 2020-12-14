from keras.applications.inception_v3 import InceptionV3
import os
import numpy as np
from tensorflow import keras
from keras import backend as K
from keras.layers import Flatten, Dense, AveragePooling2D
from keras.models import Model
from keras.optimizers import RMSprop, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.preprocessing.image import ImageDataGenerator

#from google.colab import drive
#drive.mount('/content/drive/')


# The callback below can be set in the final Model.fit() function to create Tensorboard log files.
# Note that this may dramatically slow down training time, depending on disk write speed.

#from datetime import datetime
#logdir = '/RootPathToFolder/logs/' + datetime.now().strftime("%Y%m%d-%H%M%S")
#tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)


initial_learning_rate = 0.001
img_width = 299
img_height = 299
nbr_train_samples = 5760
nbr_validation_samples = 1440
nbr_epochs = 35
batch_size = 6

# This filepath should point directly to the folder containing folders with
# sorted training data (i.e. folders named calling, normal, etc.).
# The data will be split into validation and training sets automatically by the
# image data generator later.
main_data_dir = r'/RootPathToFolder/train'

CatNames = ['calling', 'normal', 'smoking', 'smoking_calling']


# We initialize a cosine decay learning-rate scheduler for later
warmup_epochs = 10
warmup_batches = warmup_epochs * nbr_train_samples / batch_size
warmup_steps = int(warmup_epochs * nbr_train_samples / batch_size)
cosine_decay = keras.experimental.CosineDecay(initial_learning_rate, warmup_steps)

print('Loading InceptionV3 Weights...')
# Initialize InceptionV3 model
# We keep all of the default settings, except for setting include_top to False
# This removes the fully-connected top layer, so we have to specify an input shape.
# InceptionV3 can accept inputs of up to 299 on each dimension in three layers, so we use that.
InceptionV3_notop = InceptionV3(include_top = False, input_shape = (299, 299, 3))

print('Adding APL and Softmax output layers...')
# We first grab the output of our InceptionV3 model, then we add:
# 1. (8,8) average pooling layer; when include_top=False, InceptionV3 outputs a (8,8,2048) tensor, so this effectively flattens the data.
# 2. Flatten layer to fully vectorize the averaged data
# 3. Finally a softmax layer to perform the final classification.
# We append these in order to the end of the output layer.
output = InceptionV3_notop.get_layer(index = -1).output
output = AveragePooling2D((8, 8), strides = (8, 8), name = 'avg_pool')(output)
output = Flatten(name = 'flatten')(output)
output = Dense(4, activation = 'softmax', name = 'softmax_prediction')(output)

print('Building model, compiling, and making callbacks...')
# Finally, assemble the final model architecture from our modified output layer and our InceptionV3 model
InceptionV3_model = Model(InceptionV3_notop.input, output)
#InceptionV3_model.summary()

# We build an SGD object optimizer, which the Keras model type uses to determine gradient descent properties.
# In this case, we use momentum, which means the learning rate will "build speed" when changes result in improved results.
# Specifically, we use Nesterov momentum, which causes this "momentum" to build more slowly to avoid overshooting parameters.
# For our learning rate, we use a cosine decay to gradually decrease the learning rate over a given number of epochs.
# We use categorical crossentropy as our loss function, a common multi-label loss function for classifiers.
optimizer = SGD(learning_rate = cosine_decay, momentum = 0.9, decay = 0.0, nesterov = True)
InceptionV3_model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

# We want to save the best model we find, so we set a callback to do so, and to stop early if the model stops improving.
# Make sure to set the weights.h5 path to the same place as in the prediction script, or you'll have to move the file around!
best_model_file = '/FilePathToFolder/weights/weights.h5'
best_model = ModelCheckpoint(best_model_file, monitor='val_accuracy', verbose = 1, save_best_only = True, mode = 'max')
early_stop = EarlyStopping(monitor='val_accuracy', patience=2)

print('Building image data generators...')
# We create one ImageDataGenerator object, which converts the jpg data into a useable form.
# We rescale all the values into a range appropriate for InceptionV3 inputs,
# and designate 20% of the data for validation. We also have the data randomly
# flip some images horizontally, since we want to avoid things like right-handedness
# in the set from lending unwanted bias to the final model.
datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, validation_split=0.2)

# Then we build our training generator and validation generators using the datagen:
train_generator = datagen.flow_from_directory(
        main_data_dir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        shuffle = True,
        classes = CatNames,
        subset = 'training',
        class_mode = 'categorical')

validation_generator = datagen.flow_from_directory(
        main_data_dir,
        target_size =(img_width, img_height),
        batch_size = batch_size,
        shuffle = True,
        classes = CatNames,
        subset = 'validation',
        class_mode = 'categorical')

# Finally, we run our model:
InceptionV3_model.fit(
        train_generator,
        steps_per_epoch = None,
        epochs = nbr_epochs,
        validation_data = validation_generator,
        validation_steps = None,
        callbacks = [best_model, early_stop])
