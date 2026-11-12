#!/usr/bin/env python3
"""Turn a photo into an input tensor for mobilenetv2-7.

    tools/venv/bin/python prep.py myphoto.jpg myphoto.npz

Resize the short side to 256, centre-crop 224, RGB, divide by 255, subtract the
ImageNet mean and divide by the standard deviation. That is the preprocessing
the ONNX model does not contain, and forgetting it is the classic way to get
confident nonsense out of a classifier.

Then run the cvimodel on the npz:

    tools/venv/bin/model_runner --input myphoto.npz \
        --model models/mobilenetv2_bf16.cvimodel --output out.npz

Any photo will do. It always answers with one of the 1000 ImageNet classes, so
a picture of your dev board comes back as something confidently wrong.
"""
import sys

import numpy as np
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
SIZE = 224


def load_image(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if w < h:
        nw, nh = 256, int(round(256 * h / w))
    else:
        nw, nh = int(round(256 * w / h)), 256
    im = im.resize((nw, nh), Image.BILINEAR)
    left, top = (nw - SIZE) // 2, (nh - SIZE) // 2
    im = im.crop((left, top, left + SIZE, top + SIZE))
    x = np.asarray(im, np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    x = load_image(sys.argv[1])
    # mobilenetv2-7 calls its input "data", not "input"
    np.savez(sys.argv[2], data=x)
    print(f"{sys.argv[1]} -> {sys.argv[2]}  shape={x.shape} "
          f"range=[{x.min():.2f}, {x.max():.2f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
