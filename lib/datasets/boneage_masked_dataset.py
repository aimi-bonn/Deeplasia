from os.path import splitext
from os import listdir
import logging
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image
import pandas as pd
from scipy import ndimage
from torchvision import transforms
import matplotlib.pyplot as plt


def post_process(img):
    ## Remove everything except for the largest component, using connected components
    #img = (img.squeeze()).numpy()
    img = img.astype(np.uint8)
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)  # might test with 4..

    # Remove the background as a component
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    out_img = np.zeros((img.shape), dtype=np.float32)
    for i in range(0, nb_components):
        if sizes[i] >= max(sizes):
            out_img[output == i + 1] = 1.

    # Fill holes
    out_img = ndimage.binary_fill_holes(out_img).astype(np.float32)
    out_img = torch.from_numpy(out_img)
    return out_img


class BoneAgeMaskedDataset(Dataset):
    def __init__(self, imgs_dir=-1, target_file=-1, masks_dir=-1, size=512, validation=None):
        self.validation = validation

        if imgs_dir == -1:
            self.imgs_dir = "D:/Documents/Workspace/data/BoneAge/boneage-training-dataset/"
            self.target_file = "D:/Documents/Workspace/data/BoneAge/boneage-training-dataset.csv"
            self.masks_dir = "D:/Documents/Workspace/data/BoneAge/boneage-training-masks/"
        else:
            self.imgs_dir = imgs_dir
            self.target_file = target_file
            self.masks_dir = masks_dir

        self.target_file = pd.read_csv(self.target_file)
        self.clean_target_file()

        # If a validation is not given, we read and clean the target file
        if validation is None:
            self.ids = [splitext(file) for file in listdir(self.imgs_dir)
                        if not file.startswith('.')]
        # Otherwise, we use the given validation set as self.ids
        else:
            self.ids = validation
        self.ids.sort()

        logging.info(f'Creating dataset with {len(self.ids)} samples')

        self.augment_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.size = size

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

    def preprocess(self, img, mask):
        w,h = img.size

        mask_np = np.asarray(mask)
        img_np = (mask_np * np.asarray(img))

        #Create a more meaningful bounding box around the hand (mask) from PIL image
        def get_boundingbox_numpy(mask):
            # This might be faster ...
            mask_np = np.array(mask)

            maskx = np.any(mask_np, axis=0)
            masky = np.any(mask_np, axis=1)
            x1 = np.argmax(maskx)
            y1 = np.argmax(masky)
            x2 = len(maskx) - np.argmax(maskx[::-1])
            y2 = len(masky) - np.argmax(masky[::-1])
            # sub_image = image[y1:y2, x1:x2]
            return (x1,y1, x2,y2)

        # We always take the smaller image during validation, but during training some times we don't
        if self.validation is not None or random.random() > 0.75:
            # numpy: Get bounding box coords of the masked image (plus some padding around the edges)
            x1, y1, x2, y2 = get_boundingbox_numpy(mask_np)
            img_np = img_np[max(0, y1 - 50):min(h - 1, y2 + 51), max(0, x1 - 50):min(w - 1, x2 + 51)]

        # Get all non-mask pixels (background)
        mask_bg = (img_np == 0)
        mask_fg = ~mask_bg

        # We will need these values later to normalize or standardize the image
        if self.validation is not None:
            fg_min = img_np[mask_fg].min()
            fg_mean = img_np[mask_fg].mean()
            fg_max = img_np[mask_fg].max()
            fg_std = img_np[mask_fg].std()
            #bg_color = int(fg_mean)
            bg_color = 0
            img_np[mask_bg] = bg_color

        img = Image.fromarray(np.uint8(img_np))

        # Only augment if not the validation set
        if self.validation is None:

            # # Random zooming/cropping by [-0.2, .., +0.2] (double)
            # if random.random() > 0.2:
            #     max_dist = 0.2
            #     factor = 1 #(random.random() - 0.5) * 2 * max_dist     # crop range is [-max_dist, +max_dist]
            #     w_new, h_new = img.width*factor, img.height*factor
            #     img = transforms.CenterCrop((h_new, w_new))(img)

            if random.random() > 0.25:
                img = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)(img)

            # Get pixel value info from relevant (foreground) pixels
            # and afterwards set the background pixels to the mean
            img_np = np.asarray(img)

            fg_min = img_np[mask_fg].min()
            fg_mean = img_np[mask_fg].mean()
            fg_max = img_np[mask_fg].max()
            fg_std = img_np[mask_fg].std()
            bg_color = 0
            #bg_color = int(fg_mean)
            img_np[mask_bg] = bg_color

            img = Image.fromarray(np.uint8(img_np))

            ## From here on the mask might no longer match the image!
            if random.random() > 0.5:
                img = TF.hflip(img)

            if random.random() > 0.05:  # Dumb ... should be higher!
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

    def __getitem__(self, i):
        idx = self.ids[i][0]
        img_file = self.imgs_dir + idx + self.ids[i][1]
        mask_file = self.masks_dir + idx + '_resize_post.gif'

        img = Image.open(img_file)
        mask = Image.open(mask_file)
        img = self.preprocess(img, mask)

        gender = int(self.target_file.loc[int(idx)]['male'])

        # We would like to have the boneage normalized in range [-1,1] or [0,1], it usually increases model performance
        boneage = self.target_file.loc[int(idx)]['boneage']

        m, std = self.get_mean_std()
        boneage = (boneage-m) / std     # in training set, min = 1, max = 228

        return img, boneage, gender
