import numpy as np
import torch
import os
import random
from torch.utils.data import Dataset
import torch.nn.functional as F
import json
import SimpleITK as sitk
from lungsegitk.src.lung_segmentor_itk.lung_segmentor import getLungMask
import cv2

def _floodFill(segments):
    segments = segments.astype('uint8')
    def inner_floodFill(segment):
        seg_mask = segment.copy()
        cv2.floodFill(seg_mask, None, (0, 0), 1, 0, 0, 4)
        segment = cv2.bitwise_not(seg_mask * 255) + segment * 255
        return segment / 255

    z, y, x = segments.shape
    for index in range(z):
        segment = segments[index, :, :]
        segments[index, :, :] = inner_floodFill(segment)

    return segments

def load_file(data_root):
    file_list = os.listdir(data_root)
    file_list.sort()
    file_lung_dic = {}
    for f in file_list:
        ori_img = sitk.ReadImage(os.path.join(data_root, f))
        tmp_img = sitk.GetArrayFromImage(ori_img).astype('float32')

        # get lung
        lung_img = getLungMask(ori_img)[0]
        lung_array = sitk.GetArrayFromImage(lung_img)[:, :, :]
        tmp_lung = _floodFill(lung_array).astype('bool')

        ori_z, ori_y, ori_x = tmp_img.shape
        lung_z, lung_y, lung_x = np.where(tmp_lung)
        lung_z_s, lung_z_e = lung_z.min(), min(lung_z.max() + 1, ori_z)
        lung_y_s, lung_y_e = lung_y.min(), min(lung_y.max() + 1, ori_y)
        lung_x_s, lung_x_e = lung_x.min(), min(lung_x.max() + 1, ori_x)
        file_lung_dic[f] = [lung_z_s, lung_z_e, lung_y_s, lung_y_e, lung_x_s, lung_x_e]

    return file_list, file_lung_dic

class SegValCropData(Dataset):
    def __init__(self, data_root, batch_size, cube_size=128, step=64):
        self.file_list, self.file_lung_dic = load_file(data_root)
        self.root = data_root
        self.cube_size = cube_size
        self.batch_size = batch_size
        self.step = step
        self.file_dic, self.pos_list = self.crop_pos()
        self.last_name = ''
        self.img = None

    def __len__(self):
        return len(self.file_list)

    def crop_pos(self):
        file_dic = {}
        cube_size, step = self.cube_size, self.step
        for f in self.file_list:
            tmp = []
            img = sitk.ReadImage(os.path.join(self.root, f))
            x = sitk.GetArrayFromImage(img)
            lung_pos = self.file_lung_dic[f]
            # lung seg
            zl, zr, yl, yr, xl, xr = lung_pos
            x = x[zl:zr, yl:yr, xl:xr]

            x = x[np.newaxis, ...]
            xnum = (x.shape[1] - cube_size) // step + 1 if (x.shape[1] - cube_size) % step == 0 else \
                (x.shape[1] - cube_size) // step + 2
            ynum = (x.shape[2] - cube_size) // step + 1 if (x.shape[2] - cube_size) % step == 0 else \
                (x.shape[2] - cube_size) // step + 2
            znum = (x.shape[3] - cube_size) // step + 1 if (x.shape[3] - cube_size) % step == 0 else \
                (x.shape[3] - cube_size) // step + 2
            for xx in range(xnum):
                xl = step * xx
                xr = step * xx + cube_size
                if xr > x.shape[1]:
                    xr = x.shape[1]
                    xl = x.shape[1] - cube_size
                for yy in range(ynum):
                    yl = step * yy
                    yr = step * yy + cube_size
                    if yr > x.shape[2]:
                        yr = x.shape[2]
                        yl = x.shape[2] - cube_size
                    for zz in range(znum):
                        zl = step * zz
                        zr = step * zz + cube_size
                        if zr > x.shape[3]:
                            zr = x.shape[3]
                            zl = x.shape[3] - cube_size
                        tmp.append([xl, xr, yl, yr, zl, zr])
            while (len(tmp) % self.batch_size) != 0:
                tmp.append(tmp[0])
            file_dic[f] = tmp

        file_list, pos_list = [], []
        for f in self.file_list:
            file_list += [f for i in range(len(file_dic[f]))]
            pos_list += file_dic[f]
        self.file_list = file_list
        return file_dic, pos_list

    def crop(self, data, pos):
        xl, xr, yl, yr, zl, zr = pos[0], pos[1], pos[2], pos[3], pos[4], pos[5]
        data_crop = data[:, xl:xr, yl:yr, zl:zr]
        return data_crop

    def process_imgmsk(self, data):
        data = data.astype(float)
        data2 = data.copy()
        data2[data2 > 500] = 500
        data2[data2 < -1000] = -1000
        data2 = (data2 + 1000) / 1500
        data[data > 1024] = 1024
        data[data < -1024] = -1024
        data = (data + 1024) / 2048

        return data, data2

    def __getitem__(self, item):
        name = self.file_list[item]
        if name != self.last_name:
            img = sitk.ReadImage(os.path.join(self.root, name))
            img = sitk.GetArrayFromImage(img)

            lung_pos = self.file_lung_dic[name]
            zl, zr, yl, yr, xl, xr = lung_pos
            img = img[zl:zr, yl:yr, xl:xr]

            img, img2 = self.process_imgmsk(img)
            img = np.array([img, img2])
            self.img = img
            self.last_name = name
        else:
            img = self.img
        img_crop = self.crop(img, self.pos_list[item])
        pos = np.array(self.pos_list[item])

        return torch.from_numpy(img_crop.astype(np.float32)), name, torch.from_numpy(pos)