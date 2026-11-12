#!/usr/bin/env python3
"""Print the top-5 classes from a mobilenetv2 output npz.

    tools/venv/bin/python top5.py dog_out.npz

Runs on your laptop with the toolchain's python (it needs numpy). The board
has no numpy, which is why this is not part of npurun.py.
"""
import sys

import numpy as np

labels = [l.strip() for l in open("inputs/imagenet_classes.txt")]

path = sys.argv[1] if len(sys.argv) > 1 else "dog_out.npz"
d = np.load(path)

# model_runner writes every tensor it touched; the answer is the fp32 output.
keys = [k for k in d.files if k.endswith("_f32") and d[k].size == 1000]
if not keys:
    print(f"no 1000-element fp32 tensor in {path}; keys were:")
    for k in d.files:
        print(f"  {k} {d[k].shape}")
    raise SystemExit(1)

y = d[keys[-1]].reshape(-1)
for i in np.argsort(y)[::-1][:5]:
    print(f"{i:4d}  {labels[i]:32s} {y[i]:9.3f}")
