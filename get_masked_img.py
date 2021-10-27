from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from scipy import ndimage
import cv2
from PIL import Image

from torchvision import transforms
import matplotlib.pyplot as plt


augment_transform = transforms.Compose([
    transforms.ToTensor(),
])
resize = transforms.Resize((512,512))
gray = transforms.Grayscale()


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


def get_img(id, mask):
    idx = ids[id]
    img_file = imgs_dir + idx[0] + idx[1]

    img = Image.open(img_file)
    img = preprocess(img, mask)

    return img


def get_mask(id, suffix=''):
    idx = ids[id]

    mask_file = masks_dir + idx[0] + suffix + idx[1]

    mask = Image.open(mask_file)
    mask = np.asarray(mask, dtype=np.uint8)
    mask = (mask > 0.5).astype(np.uint8)
    mask_pp = post_process(np.asarray(mask))

    return mask_pp


def preprocess(img, mask):
    mask_np = np.asarray(mask)
    img_np = (mask_np * np.asarray(img))
    img = Image.fromarray(np.uint8(img_np))

    return img


imgs_dir = "D:/Documents/Workspace/data/BoneAge/boneage-training-dataset/"
target_file = "D:/Documents/Workspace/data/BoneAge/boneage-training-dataset.csv"
masks_dir = "D:/Documents/Workspace/data/BoneAge/boneage-training-masks/"

ids = [splitext(file) for file in listdir(imgs_dir) if not file.startswith('.')]

target_dir = 'out/'
suffix = '_resize'

to_get_mask = False
to_save_mask = False
to_apply_mask = True
to_save_masked_img = True
for idx, name in enumerate(ids):
    print(f"{idx=}, {name=}")
    # get_img(idx)
    if to_get_mask:
        mask = transforms.ToPILImage()(get_mask(idx, suffix=suffix))
        if to_save_mask:
            mask.save(f"{target_dir}{name[0]}{suffix}_post.gif")
    else:
        mask = Image.open(f"{target_dir}{name[0]}{suffix}_post.gif")

    if to_apply_mask:
        masked_img = get_img(idx, mask)
        if to_save_masked_img:
            masked_img.save(f"masked/{name[0]}_masked{name[1]}")