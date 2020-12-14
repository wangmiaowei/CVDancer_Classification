from tensorflow import keras
from keras.applications.inception_v3 import InceptionV3
import os
from keras.layers import Flatten, Dense, AveragePooling2D
from keras.models import Model
from keras.optimizers import RMSprop, SGD
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

from keras import backend as K
from warmup import WarmUpCosineDecayScheduler

from keras.models import load_model
import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

img_width = 299
img_height = 299
batch_size = 1
nbr_test_samples = 1710
nbr_augmentation = 5

FishNames = ['calling', 'normal', 'smoking', 'smoking_calling']

#root_path = '/Users/pengpai/Desktop/python/DeepLearning/Kaggle/NCFM'
root_path = r'C:\Users\James\Documents\The project Miaowei Wang has done\Kaggle_NCFM-master\saved_models'
weights_path = os.path.join(root_path, 'weights.h5')
test_data_dir ='C:/Users/James/Documents/The project Miaowei Wang has done/Kaggle_NCFM-master/test'
print(test_data_dir)
#D:/programs/Kaggle_NCFM-master\test/testA/
#test_data_dir ="D:\\programs\\Kaggle_NCFM-master\\test\\testA\\"
# test data generator for prediction
test_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)


print('Loading model and weights from training process ...')
InceptionV3_model = load_model(weights_path)

for idx in range(nbr_augmentation):
    print('{}th augmentation for testing ...'.format(idx))
    random_seed = np.random.random_integers(0, 100000)

    test_generator = test_datagen.flow_from_directory(
            test_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            shuffle = False, # Important !!!
            seed = random_seed,
            classes = None,
            class_mode = None)

    test_image_list = test_generator.filenames
    print('image_list: {}'.format(test_image_list[:10]))
    print('Begin to predict for testing data ...')
    if idx == 0:
        predictions = InceptionV3_model.predict_generator(test_generator, nbr_test_samples)
    else:
        predictions += InceptionV3_model.predict_generator(test_generator, nbr_test_samples)

predictions /= nbr_augmentation

print('Begin to write submission file ..')
f_submit = open(os.path.join(root_path, 'submit.csv'), 'w')
f_submit.write('image,calling,normal,smoking,smoking_calling\n')
for i, image_name in enumerate(test_image_list):
    pred = ['%.6f' % p for p in predictions[i, :]]
    if i % 100 == 0:
        print('{} / {}'.format(i, nbr_test_samples))
    f_submit.write('%s,%s\n' % (os.path.basename(image_name), ','.join(pred)))

f_submit.close()

print('Submission file successfully generated!')
