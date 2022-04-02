"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html
from util.util import save_image
from tqdm import tqdm

opt = TestOptions().parse()

dataloader = data.create_dataloader(opt)

model = Pix2PixModel(opt)
model.eval()

visualizer = Visualizer(opt)

# test
total = dataloader.dataset.dataset_size

for i, data_i in enumerate(dataloader):
    if i * opt.batchSize % 1000 == 0:
        print("processing:", i * opt.batchSize, "/", total)

    if i * opt.batchSize >= opt.how_many:
        break

    generated = model(data_i, mode='inference')

    img_path = data_i['path']
    save_path = []
    for j, path in enumerate(img_path):
        path_ = path.split("/")
        save = [opt.ouput_dir+opt.style_method+"_"+str(opt.mask_num)] + path_[2:]
        save_path.append("/".join(save))

    for b in range(generated.shape[0]):
        visuals = OrderedDict([('synthesized_image', generated[b])])
        image_numpy = visualizer.convert_visuals_to_numpy(visuals)['synthesized_image']
        save_image(image_numpy, save_path[b], create_dir=True)
