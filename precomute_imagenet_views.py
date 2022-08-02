#!/usr/bin/env python

import argparse
import numpy as np
import json
import math
import base64
import csv
import sys

csv.field_size_limit(sys.maxsize)


# CLIP Support
import torch
import clip
from PIL import Image

import os

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, 'build')
import MatterSim


TSV_FIELDNAMES = ['scanId', 'viewpointId', 'image_w','image_h', 'vfov', 'features']
VIEWPOINT_SIZE = 36 # Number of discretized views from one viewpoint
BATCH_SIZE = 4  # Some fraction of viewpoint size - batch size 4 equals 11GB memory
GPU_ID = 0

LABEL = False

parser = argparse.ArgumentParser()
parser.add_argument('--arch', default='vit16', help='architecture')
args = parser.parse_args()

# Labels Prob
if args.arch == "vit":
    FEATURE_SIZE = 512
    MODEL = "ViT-B/32"
elif args.arch == "vit16":
    FEATURE_SIZE = 512
    MODEL = "ViT-B/16"
else:
    assert False



GRAPHS = 'connectivity/'
OUTFILE = 'img_features/CLIP-ViT-B-32-views.tsv'

path = '../views_img/'

# Simulator image parameters
WIDTH=640
HEIGHT=480
VFOV=60

def load_viewpointids():
    viewpointIds = []
    with open(GRAPHS+'scans.txt') as f:
        scans = [scan.strip() for scan in f.readlines()]
        for scan in scans:
            with open(GRAPHS+scan+'_connectivity.json') as j:
                data = json.load(j)
                for item in data:
                    if item['included']:
                        viewpointIds.append((scan, item['image_id']))
    print('Loaded %d viewpoints' % len(viewpointIds))
    return viewpointIds


def build_tsv():
    # Set up the simulator
    sim = MatterSim.Simulator()
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.setRenderingEnabled(False)
    sim.init()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(MODEL, device=device, download_root='cache/')
    state_dict = model.state_dict()
    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_layers = len(
        [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size
    embed_dim = state_dict["text_projection"].shape[1]

    count = 0
    max_prob = np.zeros(FEATURE_SIZE, dtype=np.float32)
    features_list = []
    with open(OUTFILE, 'w') as tsvfile:
        writer = csv.DictWriter(tsvfile, delimiter = '\t', fieldnames = TSV_FIELDNAMES)

        # Loop all the viewpoints in the simulator
        viewpointIds = load_viewpointids()
        for scanId,viewpointId in viewpointIds:
            viewpoint_path = path + scanId + "/" + viewpointId
            if not os.path.exists(viewpoint_path):
                print("skipping scan and viewpoint:", scanId, viewpointId)
                continue

            # Loop all discretized views from this location
            blobs = []
            for ix in range(VIEWPOINT_SIZE):
                if ix == 0:
                    sim.newEpisode(scanId, viewpointId, 0, math.radians(-30))
                elif ix % 12 == 0:
                    sim.makeAction(0, 1.0, 1.0)
                else:
                    sim.makeAction(0, 1.0, 0)

                state = sim.getState()
                assert state.viewIndex == ix

                # Transform and save generated image
                image_path = path + scanId + "/" + viewpointId + "/" + str(ix) + ".png"
                im = Image.open(image_path)
                im = np.array(im)
                blobs.append(Image.fromarray(im))

            blobs = [
                preprocess(blob).unsqueeze(0)
                for blob in blobs
            ]
            blobs = torch.cat(blobs, 0)
            blobs = blobs.to(device)

            # Run as many forward passes as necessary
            with torch.no_grad():
                features = model.encode_image(blobs).float()

            if LABEL:
                for k in range(VIEWPOINT_SIZE):
                    feature = features[k]
                    max_prob = np.maximum(max_prob, feature)
                features_list.append(features.detach().cpu().numpy())
            else:
                features = features.detach().cpu().numpy()
                writer.writerow({
                    'scanId': scanId,
                    'viewpointId': viewpointId,
                    'image_w': WIDTH,
                    'image_h': HEIGHT,
                    'vfov' : VFOV,
                    'features': base64.b64encode(features).decode(),
                })

            count += 1
            if count % 100 == 0:
                print('Processed %d / %d viewpoints' %\
                  (count, len(viewpointIds)))

        if LABEL:
            for i, (scanId, viewpointId) in enumerate(viewpointIds):
                if LABEL:
                    features = features_list[i] / max_prob
                else:
                    features = features_list[i]
                writer.writerow({
                    'scanId': scanId,
                    'viewpointId': viewpointId,
                    'image_w': WIDTH,
                    'image_h': HEIGHT,
                    'vfov' : VFOV,
                    'features': base64.b64encode(features).decode(),
                })


def read_tsv(infile):
    # Verify we can read a tsv
    in_data = []
    with open(infile, "r") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = TSV_FIELDNAMES)
        for item in reader:
            item['image_h'] = int(item['image_h'])
            item['image_w'] = int(item['image_w'])
            item['vfov'] = int(item['vfov'])
            item['features'] = np.frombuffer(base64.b64decode(item['features']),
                    dtype=np.float32).reshape((VIEWPOINT_SIZE, FEATURE_SIZE))
            in_data.append(item)
    return in_data


if __name__ == "__main__":

    build_tsv()
    data = read_tsv(OUTFILE)
    print('Completed %d viewpoints' % len(data))
