# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import scipy.misc
import torch
import os

root_dir          = "CamVid/"
label_colors_file = os.path.join(root_dir, "label_colors.txt")
label2color = {}
color2label = {}
label2index = {}
index2label = {}

n_class = 32
means = np.array([103.939, 116.779, 123.68]) / 255.

model_path = "XXXXXXXXXXXXXXXXXXXXX"
model = torch.load(model_path)
use_gpu = torch.cuda.is_available()
if use_gpu:
    model = model.cuda()
model.eval()


def parse_label():
    f = open(label_colors_file, "r").read().split("\n")[:-1]  # ignore the last empty line
    for idx, line in enumerate(f):
        label = line.split()[-1]
        color = tuple([int(x) for x in line.split()[:-1]])
        print(label, color)
        label2color[label] = color
        color2label[color] = label
        label2index[label] = idx
        index2label[idx]   = label


def test_img(img_path):
    img = scipy.misc.imread(img_path, mode='RGB')
    h, w, c = img.shape[0], img.shape[1], img.shape[2]
    val_h = int(h / 32) * 32
    val_w = w
    img = scipy.misc.imresize(img, (val_h, val_w), interp='bilinear', mode=None)

    img = img[:, :, ::-1]
    img = np.transpose(img, (2, 0, 1)) / 255.
    img[0] -= means[0]
    img[1] -= means[1]
    img[2] -= means[2]

    inputs = torch.from_numpy(img.copy()).float()
    inputs = torch.unsqueeze(inputs, 0).cuda()
    output = model(inputs)
    output = output.data.cpu().numpy()

    N, _, h, w = output.shape
    assert (N == 1)
    pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(h, w)

    pred_img = np.zeros((val_h, val_w, 3), dtype=np.float32)
    for cls in range(n_class):
        pred_inds = pred == cls
        label = index2label[cls]
        color = label2color[label]
        pred_img[pred_inds] = color
    pred_img = scipy.misc.imresize(pred_img, (h, w), interp='bilinear', mode=None)
    scipy.misc.imsave('result.png', pred_img)


parse_label()
img_path = "XXXXXXXXXXXXXXXXXXXXX"
test_img(img_path)