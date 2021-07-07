# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

# from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset

import math

class BRONCHDataset(MonoDataset):
    """Superclass for different types of BRONCH dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(BRONCHDataset, self).__init__(*args, **kwargs)

        # calculating the camera intrinsic matrix for the Ambu bronchoscope
        # https://codeyarns.com/tech/2015-09-08-how-to-compute-intrinsic-camera-matrix-for-a-camera.html
        # w = 640 (virtual images I created were 640, but Ambu videos are 560) (544 so divisible by 32 as required by monodepth2)
        w = 544
        h = 480
        x = w / 2
        y = h / 2
        fov = 85 # from aScope+4+Broncho+Regular+Datasheet.pdf
        f_x = x / np.tan((fov/2) * (math.pi / 180))
        f_y = y / np.tan((fov/2) * (math.pi / 180))

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size    
        self.K = np.array([[f_x, 0, x, 0],
                           [0, f_y, y, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        self.K[0,:] /= w # divide by original width
        self.K[1,:] /= h # divide by original height

        self.full_res_shape = (w, h)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        return False

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side), self.width, self.height)

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

class BRONCHDepthDataset(BRONCHDataset):
    """
    """
    def __init__(self, *args, **kwargs):
        super(BRONCHDepthDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        image_path = os.path.join(self.data_path, "VirtualBronchoscopies", folder, "photo-{}-{}{}".format(side, frame_index, self.img_ext))
        # print(image_path)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        # f_str = "{:010d}.png".format(frame_index)
        # depth_path = os.path.join(self.data_path, "DepthPhotos", folder, "DepthPhotos/groundtruth/image_0{}".format(self.side_map[side]), f_str)
        depth_path = os.path.join(self.data_path, "VirtualBronchoscopies", folder, "photo-{}-{}{}".format(side, frame_index, self.img_ext))
        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

