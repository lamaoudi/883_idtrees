
## Model I/O:
'''
The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each image, and should be in 0-1 range. Different images can have different sizes.

The model returns a Dict[Tensor] during training, containing the classification and regression losses for both the RPN and the R-CNN.

During inference, the model requires only the input tensors, and returns the post-processed predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as follows:

- boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values between 0 and H and 0 and W
- labels (Int64Tensor[N]): the predicted labels for each image
- scores (Tensor[N]): the scores or each prediction

# For training
images = torch.rand(4, 3, 600, 1200) # 4 images, 3RGB channels, H, W
boxes = torch.rand(4, 11, 4) # 4 images, 11 boxes per img, [x1,y1,x2,y2]
labels = torch.randint(1, 91, (4, 11)) #randint betw 1 and 91(=numclasses), 11 bboxes predicted for 4 images
images = list(image for image in images)
targets = []
for i in range(len(images)):
    d = {}
    d['boxes'] = boxes[i]
    d['labels'] = labels[i]
    targets.append(d)
output = model(images, targets)
'''

import os, sys
import numpy as np
import pandas as pd
import torch
from PIL import Image
from idtrees.utils import read_tifs

from configs import *

class TreeImagesDataset(object):
    def __init__(self,csv_file, root_dir, transforms=None, object_rec=False ,
                 datatype='rgb', instance_type = 'img'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations and image directories.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            instance_type (string): 'img' : Output one image with multiple bounding boxes as item;
                            'boxes' : Output one boxes of one tree as item
        """
        self.root_dir = root_dir
        df = pd.read_csv(root_dir + csv_file, skipinitialspace=True)
        self.df = df[df.site_id!='ESALQ'] # remove ESALQ from pipeline
        self.transforms = transforms
        self.object_rec = object_rec
        self.instance_type = instance_type
        if datatype == 'rgb':
            self.uids = self.df.rgb_path.unique()
        elif datatype == 'hsi' and self.instance_type == 'boxes':
            # Only load instances for which HSI images exists
            self.uids = self.df.uid[self.df.hsi_path.notnull()]
        else:
            raise ValueError('undefined data loading task')

    def __len__(self):
        return len(self.uids)

    def _load_im(self, path):
        try:
            # windows compatible thanks to .replace()
            #img = Image.open(img_path).convert("RGB")
            img_path = os.path.join(self.root_dir, path).replace('\\', '/')
        except:
            print("check your .csv file if all the paths are defined !!!")
            print(idx, self.root_dir, path)
        img = torch.FloatTensor(read_tifs.read_tif_to_ndarray(img_path))
        return img

    def _get_boxes(self, idx):
        """
        # Loads all pixels of one instance with labels
        """
        # Load Image
        path = self.df.hsi_path.iloc[idx]
        im = self._load_im(path)

        # Crop out box
        r_box_im = self.df.width.iloc[idx] / im.shape[-1] # Ratio of RGB im box coords to load im width (e.g., r=10 for hsi)
        box = np.array([self.df.ymin.iloc[idx], self.df.ymax.iloc[idx], self.df.xmin.iloc[idx], self.df.xmax.iloc[idx]])
        box = np.around(box / r_box_im).astype(int)

        crop_im = im[:, box[0]:box[1]+1, box[2]:box[3]+1]
        if np.any(np.array(crop_im.shape) == 0):
            print('[WARNING] Loaded box has zero shape and is sketchily inflated. TODO: skip this box with ID', idx)
            if box[0] == self.df.width.iloc[idx]/r_box_im: box[0] -= 1
            if box[2] == self.df.height.iloc[idx]/r_box_im: box[2] -= 1
            crop_im = im[:, box[0]:box[1]+1, box[2]:box[3]+1]

        target = {}
        target["labels"] = self.df.class_id.iloc[idx]
        target["uid"] = self.df.uid.iloc[idx]
        

        return crop_im, target

    def _get_im(self, idx):
        """
        # Loads one image with multiple instances with labels
        """
        # load images
        path = self.uids[idx]
        img = self._load_im(path)

        # get information of each instance (e.g., tree) in a given image.
        # Each instance has its own row in the csv file,
        # so they need to be regrouped according to their path.
        groups = self.df.groupby('rgb_path')
        instances = groups.get_group(path) # contains all instances in given image

        num_objs = len(instances)
        boxes = [0.0] * num_objs
        labels = torch.zeros((num_objs,), dtype=torch.int64)
        #extras: cannot take string
#         uid = [''] * num_objs
#         sci_name = [''] * num_objs
#         nlcd_class = [''] * num_objs
        for i in range(num_objs):
#             import pdb; pdb.set_trace()
            boxes[i] = [instances.xmin.iloc[i], instances.ymin.iloc[i],
                        instances.xmax.iloc[i], instances.ymax.iloc[i]]
#             uid[i] = self.df.uid.iloc[idx]
#             sci_name[i] = instances.scientific_name.iloc[i]
#             nlcd_class[i] = instances.nlcd_class.iloc[i]
            if self.object_rec == False:
                labels[i] = float(instances.class_id.iloc[i])

        if self.object_rec == True: # overwrite labels for object recognition task
            labels = torch.ones((num_objs,), dtype=torch.int64)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        image_id = torch.tensor([idx])
        # for pycocotools MAP evaluation metric
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        #extras: cannot take string
#         target["site_id"] = instances.site_id.iloc[0]
#         target["uid"] = uid
#         target["sci_name"] = sci_name
#         target["nlcd_class"] = nlcd_class
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __getitem__(self, idx):

        if self.instance_type == 'img':
            img, target = self._get_im(idx)
        elif self.instance_type == 'boxes':
            img, target = self._get_boxes(idx)

        return img, target
