import numpy as np
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.functional import cross_entropy, sigmoid, binary_cross_entropy
from WingsNet import WingsNet
from Data import SegValCropData
import skimage.measure as measure
import nibabel
from skimage.morphology import skeletonize_3d
import SimpleITK as sitk
import argparse
from postprocessing import postprocess, back_original_size

parser = argparse.ArgumentParser('Docker for airway segmentation by team timi')
parser.add_argument('-i', "--inputs", default='./inputs', type=str, help="input path of the CT images list")
parser.add_argument('-o', "--outputs", default='./outputs', type=str, help="output of the prediction results list")
args = parser.parse_args()

def test(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    data_root = args.inputs
    save_root = args.outputs

    model = WingsNet(in_channel=2, n_classes=1)
    valid_dataset = SegValCropData(data_root, batch_size=8, cube_size=128, step=64)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False, num_workers=1,
                                  pin_memory=True, drop_last=True)
    pos_dic = valid_dataset.file_lung_dic
    # resume
    weights_dict = torch.load(os.path.join('./model', 'wingsnet_37.pth'))
    model.load_state_dict(weights_dict, strict=False)
    model = torch.nn.DataParallel(model).cuda()
    model.train()

    last_name = ''
    with torch.no_grad():
        for i, (x, name, pos) in enumerate(valid_dataloader):
            name = name[0]
            if name != last_name:
                if last_name != '':
                    print(last_name)
                    # pred_num[pred_num == 0] = 1
                    pred = pred / pred_num
                    pred = pred[0, 0]
                    pred[pred >= 0.5] = 1
                    pred[pred < 0.5] = 0
                    pred = postprocess(pred)
                    back_original_size(pred, last_name, pos_dic, data_root, save_root)

                img = sitk.ReadImage(os.path.join(data_root, name))
                arr = sitk.GetArrayFromImage(img)
                lung_pos = pos_dic[name]
                zmin, zmax, ymin, ymax, xmin, xmax = lung_pos
                arr = arr[zmin:zmax, ymin:ymax, xmin:xmax]
                pred = np.zeros(arr.shape)
                pred = pred[np.newaxis, np.newaxis, ...]
                pred_num = np.zeros(pred.shape)
                last_name = name

            x = x.cuda()
            p0, p = model(x)
            p = torch.sigmoid(p)
            p = p.cpu().detach().numpy()
            pos = pos.numpy()
            for i in range(len(pos)):
                # print(pos)
                xl, xr, yl, yr, zl, zr = pos[i, 0], pos[i, 1], pos[i, 2], pos[i, 3], pos[i, 4], pos[i, 5]
                pred[0, :, xl:xr, yl:yr, zl:zr] += p[i]
                pred_num[0, :, xl:xr, yl:yr, zl:zr] += 1

        # pred_num[pred_num == 0] = 1
        pred = pred / pred_num
        pred = pred[0, 0]
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        print(last_name)
        pred = postprocess(pred)
        back_original_size(pred, last_name, pos_dic, data_root, save_root)

if __name__ == '__main__':
    test(args)


















