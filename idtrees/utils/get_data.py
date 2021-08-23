'''
Get a list of tensors and a list of classes.
'''
import sys

import numpy as np
import random
import csv

from idtrees.utils import read_tifs #, load_data # Import data utils
import torch
from idtrees.utils.dataset import TreeImagesDataset

import matplotlib.pyplot as plt
from configs import *

import pandas as pd



def get_data(total_classes=6):
    # Load hsi dataset of bounding boxes with torch dataloader
    csv_file ='data/data_train_mitree.csv'
    root_dir = repo_path
    image_dataset = TreeImagesDataset(csv_file, root_dir, object_rec=False, datatype='hsi', instance_type = 'boxes')
    image_dataset = torch.utils.data.Subset(image_dataset, [0]) # TODO: find out what the [0] does

    # Get number of bounding boxes in dataset
    n_boxes = image_dataset.dataset.__len__()
    print('Number of trees, labeled with species and bounding box: ', n_boxes)

    im_all = []
    class_ids = []
    for idx in range(n_boxes): 
        # Get image and target of current bounding box 
        im, target = image_dataset.dataset.__getitem__(idx)
        im_all.append(im)
        class_ids.append(target['labels'])
        
    classes = np.unique(class_ids)
    class_ids = np.array(class_ids)

    df = pd.read_csv(repo_path + 'data/train/Field/taxonID_ScientificName.csv')

    class_id_val = np.zeros(len(classes))
    n_im_val = np.zeros(len(classes))
    sci_names = []

    # Iterate over each class and print class id, number of pixels, and scientific name
    # print('cls_id \tn_px \tscientific name')
    for i, c in enumerate(classes):
        ids_in_c = np.where(class_ids == c)[0]
        n_im = ids_in_c.shape[0]
        sci_name = df[df.taxonCode==c].scientificName.iloc[0]
        class_id_val[i] = c
        n_im_val[i] = n_im
        sci_names.append(sci_name)

    keep = class_id_val[n_im_val.argsort()[-total_classes:]]
    print("Classes to be used:",keep)
    print("Counts for these classes", n_im_val[n_im_val.argsort()[-total_classes:]])

    im_keep = []
    im_not_keep = []
    new_class_ids = []


    for c in keep:
        ids_in_c = np.where(class_ids == c)[0]
        n_im = [im_all[x] for x in ids_in_c]
        for n in n_im:
            im_keep.append(n)
            new_class_ids.append(c)

    for c in classes:
        if c not in keep:
            ids_nin_c = np.where(class_ids == c)[0] 
            n_im = [im_all[x] for x in ids_nin_c]
            for n in n_im:
                im_not_keep.append(n)
                new_class_ids.append(34.)

    # print(len(im_not_keep), len(im_keep))

    #combine into new total list, with a 34th class id known as 'Other'
    im_all_new = im_keep 
    im_all_new.extend(im_not_keep)

    new_classes = np.unique(new_class_ids)
    # print(new_classes)

    df.loc[34, :] = [34., 'OTHER', 'ALL Others']

    class_id_val = []
    n_im_val = []
    sci_names = []

    # Iterate over each class and print class id, number of pixels, and scientific name
    # print('cls_id \tn_px \tscientific name')

    for c in new_classes:
        ids_in_c = np.where(new_class_ids == c)[0]
        n_im = len([new_class_ids[x] for x in ids_in_c])
        sci_name = df[df.taxonCode==c].scientificName.iloc[0]
        class_id_val.append(c)
        n_im_val.append(n_im)
        sci_names.append(sci_name)

    data = read_tifs.get_hsi_pixels()

    class_id_val = []
    n_px_val = []
    sci_names = []
    special_val_px = []

    class_id_vals = np.unique(data[0,:]) # Class_ids should start with 1

    # Iterate over each class and print class id, number of pixels, and scientific name
    # print('cls_id \tn_px \tscientific name')
    for c in class_id_vals:
        if c in keep:
            ids_in_c = np.argwhere([data[0,:] == c])[:,1]
            n_px = data[0,ids_in_c].shape[0]
            sci_name = df[df.taxonCode==c].scientificName.iloc[0]
            class_id_val.append(c)
            n_px_val.append(n_px)
            sci_names.append(sci_name)

        else: 
            ids_in_c = np.argwhere([data[0,:] == c])[:,1]
            n_px = data[0,ids_in_c].shape[0]
            special_val_px.append(n_px)

    return im_all_new, new_class_ids, class_id_val, n_px_val, sci_names, special_val_px
