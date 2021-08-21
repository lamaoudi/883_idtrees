# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__': #only for dev purposes

    labels_frame = pd.read_csv('../data/train/Field/taxonID_ScientificName.csv')
    labelsID = labels_frame.taxonCode
    labelsname = labels_frame.scientificName

    import geopandas as gpd
    bbox_MLBS = gpd.read_file("../data/train/ITC/train_MLBS.shp")
    bbox_OSBS = gpd.read_file("../data/train/ITC/train_OSBS.shp")

    bbox_OSBS.geometry
    bbox_MLBS.geometry

    

    df = pd.read_csv('/n/home00/nwendt/projects/idtrees/data/toydata.csv')
#     pf = np.('/n/home00/nwendt/projects/idtrees/data/toydata.csv')
    
    df.groupby('rgb_path')
    paths = df.rgb_path.unique()
    paths
    for path in df.rgb_path.unique():
#         print(paths.get_group(path))
        instances = paths.get_group(path)
        print(len(instances))
        num_objs = len(instances)
        print(path)
        
        
    paths = df.rgb_path.unique() # self.paths
    idx = 0
    path = paths[idx]
    groups = df.groupby('rgb_path')
    instances = groups.get_group(path)
    instances
    
    len(instances)
    boxes = []
    for i in range(len(instances)):
        boxes.append([instances.xmin[i], instances.ymin[i],
                      instances.xmax[i], instances.ymax[i] ]) # normalized yet or not ?

    instances.xmin[0]
    boxes = []
    for i in range(len(instances)):
#         print(i.xmin)
        print(instances.xmin[i])
#         boxes.append([i.xmin, i.ymin, i.xmax, i.ymax]) # normalized yet or not ?
#     print(boxes)
    
    
    paths.first()
    paths.get_group()
    gk.get_group('Boston Celtics') 
    df.rgb_path[0]
    
    root_dir = '/n/home00/nwendt/projects/'
    csv_file ='idtrees/data/toydata.csv'
    df = pd.read_csv(root_dir + csv_file)
    df
    import os
    
    img_name = os.path.join(root_dir,
                    df.rgb_path[0]).replace('\\', '/')
    image = io.imread(img_name)
    
    image
    bbox = [df.xmin[0], df.ymin[0],
                df.xmax[0], df.ymax[0], ]    
    bbox = np.asarray(bbox)        
    
class TreeImagesDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])


        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
    
    
class TreeImagesDatasetToy(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations and image directories.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.df = pd.read_csv(root_dir + csv_file, skipinitialspace=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.df.rgb_path[idx]).replace('\\', '/') #windows compatibility
        image = io.imread(img_name)
        bbox = [self.df.xmin[idx], self.df.ymin[idx],
                self.df.xmax[idx], self.df.ymax[idx], ]
        bbox = np.array(bbox).astype('float') #.reshape(-1, 2)
        label = self.df.class_code[idx]

        # already adapted by nils below
        sample = {'image': image, 'bbox': bbox}

        if self.transform:
            sample = self.transform(sample)
        
        return sample['image'], {'bbox':sample['bbox'], 'label': label}

    