from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import torchvision.transforms.functional as TF
import pandas as pd

import random

from torchvision import transforms
import matplotlib.pyplot as plt


class BoneAgeDataset(Dataset):
    def __init__(self, imgs_dir=-1, target_file=-1, size=512, validation=None):
        self.size = size
        if imgs_dir == -1:
            self.imgs_dir = "D:/Documents/Workspace/data/BoneAge/boneage-training-dataset/"
            self.target_file = "D:/Documents/Workspace/data/BoneAge/boneage-training-dataset.csv"
        else:
            self.imgs_dir = imgs_dir
            self.target_file = target_file

        self.target_file = pd.read_csv(self.target_file)
        self.clean_target_file()

        # If a validation is not given, we read and clean the target file
        if validation is None:
            self.ids = [splitext(file) for file in listdir(self.imgs_dir)
                        if not file.startswith('.')]
        # Otherwise, we use the given validation set as self.ids
        else:
            self.ids = validation

        self.validation = validation

        self.ids.sort()
        logging.info(f'Creating dataset with {len(self.ids)} examples')

        self.augment_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.ids)

    def split_and_get_val(self, ratio=0.1):
        # Randomly sample from training set to get val set
        val_idxs = np.random.choice(len(self.ids), int(ratio*len(self.ids)), replace=False)
        val_idxs = np.sort(val_idxs)[::-1]

        # Remove the values in the training set which are now placed in the val set
        val_ids = [self.ids.pop(idx) for idx in val_idxs]
        val_ids.sort()
        self.ids.sort()

        logging.info(f'Split a val set with {len(val_ids)} samples from the training set, {len(self.ids)} remaining.')

        return val_ids

    def clean_target_file(self):
        # Test csv
        if self.target_file.columns[-1] == 'Sex':
            self.target_file = self.target_file.replace('M', True)
            self.target_file = self.target_file.replace('F', False)
            self.target_file = self.target_file.rename(columns={"Case ID": "id", "Sex": "male"})

        # Validation csv
        elif self.target_file.columns[-1] == 'Bone Age (months)':
            self.target_file = self.target_file.replace('TRUE', True)
            self.target_file = self.target_file.replace('FALSE', False)
            self.target_file = self.target_file.rename(columns={"Image ID": "id", "Bone Age (months)": "boneage"})

        # Training csv
        elif self.target_file.columns[-1] == 'male':
            pass

        self.target_file = self.target_file.set_index('id')

    def get_mean_std(self):
        return self.target_file['boneage'].mean(), self.target_file['boneage'].std()

    # TODO: Replace random.random() with np or torch, random.random is NOT reproducible!
    def preprocess(self, img):
        # We will need these values later to normalize or standardize the image
        if self.validation is not None:
            img_np = np.array(img)
            fg_min = img_np.min()
            fg_mean = img_np.mean()
            fg_max = img_np.max()
            fg_std = img_np.std()
            #bg_color = int(fg_mean)
            bg_color = 0
            img = Image.fromarray(np.uint8(img_np))

        if self.validation is None:
            if random.random() > 0.25:
                img = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)(img)

            # Get pixel value info from relevant (foreground) pixels
            # and afterwards set the background pixels to the mean
            img_np = np.asarray(img)

            fg_min = img_np.min()
            fg_mean = img_np.mean()
            fg_max = img_np.max()
            fg_std = img_np.std()
            bg_color = 0
            #bg_color = int(fg_mean)

            img = Image.fromarray(np.uint8(img_np))

            # From here on the mask might no longer match the image!

            if random.random() > 0.5:
                img = TF.hflip(img)

            if random.random() > 0.05:
                img = transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.8, 1.2), #shear=0.01,
                                              fill=bg_color, interpolation=transforms.InterpolationMode.BILINEAR)(img)

        img = transforms.Resize((self.size, self.size))(img)
        img = self.augment_transform(img)

        # Standardize, based on foreground pixel values
        # note: we divide fg-values by 255 as it is now in tensor format [0..1]
        #img = (img - fg_mean/255) / (fg_std/255 + 1e-8)
        img = (img - img.mean()) / (img.std() + 1e-8)

        # Normalize, based on foreground pixel values
        #img = (img - fg_min/255) / ((fg_max - fg_min)/255 + 1e-8)

        # plt.imshow(img.permute(1,2,0))
        # plt.show()

        # if img.max() > 1:
        #     img = img / 255

        return img

    def __getitem__(self, i, to_augment=True):
        idx = self.ids[i][0]
        img_file = self.imgs_dir + idx + self.ids[i][1]

        img = Image.open(img_file)
        img = self.preprocess(img)

        gender = int(self.target_file.loc[int(idx)]['male'])

        # We would like to have the boneage normalized in range [-1,1] or [0,1], it usually increases model performance
        boneage = self.target_file.loc[int(idx)]['boneage']

        m, std = self.get_mean_std()
        boneage = (boneage-m) / std # min in training set = 1, max = 228

        # return self.transform(img), self.transform(mask)
        return img, boneage, gender
