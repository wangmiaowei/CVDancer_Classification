import os
import numpy as np
import shutil

np.random.seed(2016)

root_train = 'D:/programs/Kaggle_NCFM-master/data/train_split'
root_val = 'D:/programs/Kaggle_NCFM-master/data/val_split'

root_total = r'D:\programs\Kaggle_NCFM-master\data\train'

FishNames = ['calling', 'normal', 'smoking', 'smoking_calling']

nbr_train_samples = 0
nbr_val_samples = 0

# Training proportion
split_proportion = 0.8

for fish in FishNames:
    if fish not in os.listdir(root_train):
        os.mkdir(os.path.join(root_train, fish))

    total_images = os.listdir(os.path.join(root_total, fish))

    nbr_train = int(len(total_images) * split_proportion)

    np.random.shuffle(total_images)

    train_images = total_images[:nbr_train]

    val_images = total_images[nbr_train:]

    for img in train_images:
        source = os.path.join(root_total, fish, img)
        target = os.path.join(root_train, fish, img)
        shutil.copy(source, target)
        nbr_train_samples += 1

    if fish not in os.listdir(root_val):
        os.mkdir(os.path.join(root_val, fish))

    for img in val_images:
        source = os.path.join(root_total, fish, img)
        target = os.path.join(root_val, fish, img)
        shutil.copy(source, target)
        nbr_val_samples += 1

print('Finish splitting train and val images!')
print('# training samples: {}, # val samples: {}'.format(nbr_train_samples, nbr_val_samples))
