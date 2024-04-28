import argparse
import numpy as np
import random
from glob import glob

import torch

from PIL import Image
from ram.models import ram_plus
from ram import inference_ram as inference
from ram import get_transform
from tqdm import tqdm


parser = argparse.ArgumentParser(
    description='Tag2Text inferece for tagging and captioning')
parser.add_argument('--image',
                    metavar='DIR',
                    help='path to dataset')
parser.add_argument('--pretrained',
                    metavar='DIR',
                    help='path to pretrained model',
                    default='pretrained/ram_plus_swin_large_14m.pth')
parser.add_argument('--image-size',
                    default=384,
                    type=int,
                    metavar='N',
                    help='input image size (default: 448)')


if __name__ == "__main__":

    args = parser.parse_args()

    print("CUDA: ", torch.cuda.is_available())
    print("MPS: ", torch.backends.mps.is_available())
    device = torch.device(
            'cuda' if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(device)

    transform = get_transform(image_size=args.image_size)

    #######load model
    model = ram_plus(pretrained=args.pretrained,
                             image_size=args.image_size,
                             vit='swin_l')
    model.eval()

    model = model.to(device)

    with open("tags.csv", "w") as f:
        for path in tqdm(glob(args.image + "*/*.*")):
            if not (path.endswith("jpg") or path.endswith("png")):
                continue
            f.write(path + " | ")
            try:
                image = transform(Image.open(path)).unsqueeze(0).to(device)

                res = inference(image, model)
            except OSError as exc:
                tqdm.write("{path} failed with {exc}\n".format(path=path, exc=exc))
            f.write(res[0] + "\n")
