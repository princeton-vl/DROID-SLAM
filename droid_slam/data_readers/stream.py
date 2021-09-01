
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import csv
import os
import cv2
import math
import random
import json
import pickle
import os.path as osp

from .rgbd_utils import *

class RGBDStream(data.Dataset):
    def __init__(self, datapath, frame_rate=-1, image_size=[384,512], crop_size=[0,0]):
        self.datapath = datapath
        self.frame_rate = frame_rate
        self.image_size = image_size
        self.crop_size = crop_size
        self._build_dataset_index()
    
    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

    @staticmethod
    def depth_read(depth_file):
        return np.load(depth_file)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """ return training video """
        image = self.__class__.image_read(self.images[index])
        image = torch.from_numpy(image).float()
        image = image.permute(2, 0, 1)

        try:
            tstamp = self.tstamps[index]
        except:
            tstamp = index

        pose = torch.from_numpy(self.poses[index]).float()
        intrinsic = torch.from_numpy(self.intrinsics[index]).float()

        # resize image
        sx = self.image_size[1] / image.shape[2]
        sy = self.image_size[0] / image.shape[1]

        image = F.interpolate(image[None], self.image_size, mode='bilinear', align_corners=False)[0]

        fx, fy, cx, cy = intrinsic.unbind(dim=0)
        fx, cx = sx * fx, sx * cx
        fy, cy = sy * fy, sy * cy
        
        # crop image
        if self.crop_size[0] > 0:
            cy = cy - self.crop_size[0]
            image = image[:,self.crop_size[0]:-self.crop_size[0],:]

        if self.crop_size[1] > 0:
            cx = cx - self.crop_size[1]
            image = image[:,:,self.crop_size[1]:-self.crop_size[1]]

        intrinsic = torch.stack([fx, fy, cx, cy])

        return tstamp, image, pose, intrinsic


class ImageStream(data.Dataset):
    def __init__(self, datapath, intrinsics, rate=1, image_size=[384,512]):
        rgb_list = osp.join(datapath, 'rgb.txt')
        if os.path.isfile(rgb_list):
            rgb_list = np.loadtxt(rgb_list, delimiter=' ', dtype=np.unicode_)
            self.timestamps = rgb_list[:,0].astype(np.float)
            self.images = [os.path.join(datapath, x) for x in rgb_list[:,1]]
            self.images = self.images[::rate]
            self.timestamps = self.timestamps[::rate]

        else:
            import glob
            self.images = sorted(glob.glob(osp.join(datapath, '*.jpg'))) +  sorted(glob.glob(osp.join(datapath, '*.png')))
            self.images = self.images[::rate]

        self.intrinsics = intrinsics
        self.image_size = image_size

    def __len__(self):
        return len(self.images)

    @staticmethod
    def image_read(imfile):
        return cv2.imread(imfile)

    def __getitem__(self, index):
        """ return training video """
        image = self.__class__.image_read(self.images[index])

        try:
            tstamp = self.timestamps[index]
        except:
            tstamp = index

        ht0, wd0 = image.shape[:2]
        ht1, wd1 = self.image_size

        intrinsics = torch.as_tensor(self.intrinsics)
        intrinsics[0] *= wd1 / wd0
        intrinsics[1] *= ht1 / ht0
        intrinsics[2] *= wd1 / wd0
        intrinsics[3] *= ht1 / ht0

        # resize image
        ikwargs = {'mode': 'bilinear', 'align_corners': True}
        image = torch.from_numpy(image).float().permute(2, 0, 1)
        image = F.interpolate(image[None], self.image_size, **ikwargs)[0]

        return tstamp, image, intrinsics



class StereoStream(data.Dataset):
    def __init__(self, datapath, intrinsics, rate=1, image_size=[384,512], 
            map_left=None, map_right=None, left_root='image_left', right_root='image_right'):
        import glob
        self.intrinsics = intrinsics
        self.image_size = image_size
        
        imgs = sorted(glob.glob(osp.join(datapath, left_root, '*.png')))[::rate]
        self.images_l = []
        self.images_r = []
        self.tstamps = []

        for img_l in imgs:
            img_r = img_l.replace(left_root, right_root)
            if os.path.isfile(img_r):
                t = np.float(img_l.split('/')[-1].replace('.png', ''))
                self.tstamps.append(t)
                self.images_l += [ img_l ]
                self.images_r += [ img_r ]

        self.map_left = map_left
        self.map_right = map_right

    def __len__(self):
        return len(self.images_l)

    @staticmethod
    def image_read(imfile, imap=None):
        image = cv2.imread(imfile)
        if imap is not None:
            image = cv2.remap(image, imap[0], imap[1], interpolation=cv2.INTER_LINEAR)
        return image

    def __getitem__(self, index):
        """ return training video """
        tstamp = self.tstamps[index]
        image_l = self.__class__.image_read(self.images_l[index], self.map_left)
        image_r = self.__class__.image_read(self.images_r[index], self.map_right)

        ht0, wd0 = image_l.shape[:2]
        ht1, wd1 = self.image_size

        intrinsics = torch.as_tensor(self.intrinsics)
        intrinsics[0] *= wd1 / wd0
        intrinsics[1] *= ht1 / ht0
        intrinsics[2] *= wd1 / wd0
        intrinsics[3] *= ht1 / ht0

        image_l = torch.from_numpy(image_l).float().permute(2, 0, 1)
        image_r = torch.from_numpy(image_r).float().permute(2, 0, 1)

        # resize image
        ikwargs = {'mode': 'bilinear', 'align_corners': True}
        image_l = F.interpolate(image_l[None], self.image_size, **ikwargs)[0]
        image_r = F.interpolate(image_r[None], self.image_size, **ikwargs)[0]

        return tstamp, image_l, image_r, intrinsics



# class RGBDStream(data.Dataset):
#     def __init__(self, datapath, intrinsics=None, rate=1, image_size=[384,512]):
#         assoc_file = osp.join(datapath, 'associated.txt')
#         assoc_list = np.loadtxt(assoc_file, delimiter=' ', dtype=np.unicode_)
        
#         self.intrinsics = intrinsics
#         self.image_size = image_size
        
#         self.timestamps = assoc_list[:,0].astype(np.float)[::rate]
#         self.images = [os.path.join(datapath, x) for x in assoc_list[:,1]][::rate]
#         self.depths = [os.path.join(datapath, x) for x in assoc_list[:,3]][::rate]

#     def __len__(self):
#         return len(self.images)

#     @staticmethod
#     def image_read(imfile):
#         return cv2.imread(imfile)

#     @staticmethod
#     def depth_read(depth_file):
#         depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
#         return depth.astype(np.float32) / 5000.0

#     def __getitem__(self, index):
#         """ return training video """
#         tstamp = self.timestamps[index]
#         image = self.__class__.image_read(self.images[index])
#         depth = self.__class__.depth_read(self.depths[index])

#         ht0, wd0 = image.shape[:2]
#         ht1, wd1 = self.image_size

#         intrinsics = torch.as_tensor(self.intrinsics)
#         intrinsics[0] *= wd1 / wd0
#         intrinsics[1] *= ht1 / ht0
#         intrinsics[2] *= wd1 / wd0
#         intrinsics[3] *= ht1 / ht0

#         # resize image
#         ikwargs = {'mode': 'bilinear', 'align_corners': True}
#         image = torch.from_numpy(image).float().permute(2, 0, 1)
#         image = F.interpolate(image[None], self.image_size, **ikwargs)[0]

#         depth = torch.from_numpy(depth).float()[None,None]
#         depth = F.interpolate(depth, self.image_size, mode='nearest').squeeze()

#         return tstamp, image, depth, intrinsics
