from keras.applications.inception_v3 import InceptionV3
import os
from keras.layers import Flatten, Dense, AveragePooling2D
from keras.models import Model
from keras.optimizers import RMSprop, SGD
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow import keras
from keras import backend as K
from warmup import WarmUpCosineDecayScheduler
learning_rate = 0.0001
img_width = 299
img_height = 299
nbr_train_samples =  5760
nbr_validation_samples = 1440
nbr_epochs = 35
batch_size = 6

train_data_dir = r'D:\programs\Kaggle_NCFM-master\data\train_split'
val_data_dir = r'D:\programs\Kaggle_NCFM-master\data\val_split'

FishNames = ['calling', 'normal', 'smoking', 'smoking_calling']
# Compute the number of warmup batches.
warmup_epoch = 10
sample_count = 5760
warmup_batches = warmup_epoch * sample_count / batch_size
learning_rate_base = 0.0001
total_steps = int(nbr_epochs * sample_count / batch_size)
warmup_steps = int(warmup_epoch * sample_count / batch_size)
# Create the Learning rate scheduler.
warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                            total_steps=total_steps,
                                            warmup_learning_rate=4e-06,
                                            warmup_steps=warmup_steps,
                                            hold_base_rate_steps=5,
                                            )
print('Loading InceptionV3 Weights ...')
InceptionV3_notop = InceptionV3(include_top=False, weights='imagenet',
                    input_tensor=None, input_shape=(299, 299, 3))
# Note that the preprocessing of InceptionV3 is:
# (x / 255 - 0.5) x 2

print('Adding Average Pooling Layer and Softmax Output Layer ...')
output = InceptionV3_notop.get_layer(index = -1).output  # Shape: (8, 8, 2048)
output = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(output)
output = Flatten(name='flatten')(output)
output = Dense(4, activation='softmax', name='predictions')(output)

InceptionV3_model = Model(InceptionV3_notop.input, output)
#InceptionV3_model.summary()

optimizer = SGD(lr = learning_rate, momentum = 0.9, decay = 0.0, nesterov = True)
InceptionV3_model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

# autosave best Model
best_model_file = "D:/programs/Kaggle_NCFM-master/weights.h5"
best_model = ModelCheckpoint(best_model_file, monitor='val_acc', verbose = 1, save_best_only = True,mode=max)

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        rotation_range=10.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)

# this is the augmentation configuration we will use for validation:
# only rescaling
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        shuffle = True,
        # save_to_dir = '/Users/pengpai/Desktop/python/DeepLearning/Kaggle/NCFM/data/visualization',
        # save_prefix = 'aug',
        classes = FishNames,
        class_mode = 'categorical')

validation_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle = True,
        #save_to_dir = '/Users/pengpai/Desktop/python/DeepLearning/Kaggle/NCFM/data/visulization',
        #save_prefix = 'aug',
        classes = FishNames,
        class_mode = 'categorical')

InceptionV3_model.fit_generator(
        train_generator,
        samples_per_epoch = nbr_train_samples,
        nb_epoch = nbr_epochs,
        validation_data = validation_generator,
        nb_val_samples = nbr_validation_samples,
        callbacks = [warm_up_lr,best_model])
save_dir= "D:/programs/Kaggle_NCFM-master/saved_models/"
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, "weights.h5")
InceptionV3_model.save(model_path)
print('Saved trained model at %s ' % model_path)