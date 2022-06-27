import os

from PIL import Image
import numpy as np

import json

import argparse


def semantic_transfer(args):
    with open("label2index.json") as f:
        label2index = json.load(f)
    f.close()

    path = args.input
    semantic_path = args.output

    with open("label2color.json") as f:
        label2color = json.load(f)

    color2label = dict()
    for label, color in label2color.items():
        r = color["R"]
        g = color["G"]
        b = color["B"]
        color2label[(r,g,b)] = label

    with open("scandict.json") as f:
        scan_dict = json.load(f)
    f.close()

    for j, scan in enumerate(scan_dict):
        if os.path.isdir(path + "/" + scan):
            print("scan:", scan, "progress:", j, "/", len(scan_dict))
            views = os.listdir(path + "/" + scan)
            for view in views:
                print("view:", view)
                if not os.path.isdir(path + "/" + scan + "/" + view):
                    continue
                imgs = os.listdir(path + "/" + scan + "/" + view)

                for i, img in enumerate(imgs):
                    if img[-4:] != '.png':
                        print("skipping file:", img)
                        continue
                    image = Image.open(path + "/" + scan + "/" + view + "/" + img)
                    image = np.array(image)
                    w,h,_ = image.shape
                    image_semantic = np.zeros((w,h), dtype=np.uint8)
                    for k in color2label:
                        image_semantic[(image == k).all(axis=2)] = label2index[color2label[k]]
                    im = Image.fromarray(image_semantic)
                    im = im.convert("L")

                    dir = semantic_path + "/" + scan + "/" + view
                    if not os.path.exists(dir):
                        os.makedirs(dir)

                    im.save(semantic_path + "/" + scan + "/" + view + "/" + img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default="views_sem_image")
    parser.add_argument('--output', default='views_sem_image_transferred')
    args = parser.parse_args()

    semantic_transfer(args)