from styleaug import StyleAugmentor

import torch
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import os
import torchvision.transforms as transforms
import argparse

# PyTorch Tensor <-> PIL Image transforms:
toTensor = ToTensor()
toPIL = ToPILImage()


loader = transforms.Compose([
    transforms.Resize((480,640)),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def image_loader(image_name):
    image = Image.open(image_name)
    # print(image.size())
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def environment_creation(args):
    path = args.input
    output_path = args.output
    files = os.listdir(path)

    augmentor = StyleAugmentor()
    unloader = transforms.ToPILImage()

    for j, scan in enumerate(files):
        if os.path.isdir(path + "/" + scan):
            print("scan:", scan, "progress:", j, "/", len(files))
            views = os.listdir(path + "/" + scan)
            # embedding = augmentor.sample_embedding(1)                    # Sample the same style embedding for all the views in a scan
            for view in views:
                print("view:", view)
                imgs = os.listdir(path + "/" + scan + "/" + view)
                embedding = augmentor.sample_embedding(1)                  # Sample the same style embedding for all discretized views in a panorama
                for i, img in enumerate(imgs):
                    content_img = image_loader(path + "/" + scan + "/" + view + "/" + img)
                    im_restyled = augmentor(content_img, embedding=embedding)

                    image = im_restyled.squeeze(0).cpu().detach()

                    image = unloader(image)

                    dir = "%s/%s" % (output_path, scan+"/"+view)
                    if not os.path.exists(dir):
                        os.makedirs(dir)

                    image.save("%s/%s" % (output_path, scan+"/"+view+"/"+img))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
                        default="views_img")
    parser.add_argument('--output', default='views_img_style_transfer')
    args = parser.parse_args()

    environment_creation(args)
