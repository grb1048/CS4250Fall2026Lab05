#!/usr/bin/env python3
"""Generate the ONNX models this lab compiles.

    tools/venv/bin/python gen_onnx.py onnx/

Writes:
  convnet_{32,64,128,256}.onnx   the same 898-parameter convnet at four sizes
  flip_gather.onnx               a channel flip expressed as Gather
  dynamic_range.onnx             Range() whose length comes from the input

Nothing here is downloaded and nothing is trained. The weights are seeded
random numbers, which is the point: this is a model whose every shape is static
and whose every operator is on the TPU's list, so that a failure means the
toolchain is broken rather than the model is weird.
"""
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

rng = np.random.default_rng(0)


def convnet(n):
    """1x3xnxn -> 10 logits. 898 parameters, independent of n.

    Two 3x3 convolutions with 8 channels, a global average pool, a flatten and
    an 8x10 matmul. Work is 216*n^2 + 576*n^2 MACs.
    """
    W1 = (rng.standard_normal((8, 3, 3, 3)) * 0.1).astype(np.float32)
    W2 = (rng.standard_normal((8, 8, 3, 3)) * 0.1).astype(np.float32)
    W3 = (rng.standard_normal((8, 10)) * 0.1).astype(np.float32)
    B1 = np.zeros(8, np.float32)
    B2 = np.zeros(8, np.float32)
    B3 = np.zeros(10, np.float32)

    inits = [numpy_helper.from_array(a, nm) for a, nm in
             [(W1, "W1"), (B1, "B1"), (W2, "W2"), (B2, "B2"),
              (W3, "W3"), (B3, "B3")]]

    nodes = [
        helper.make_node("Conv", ["input", "W1", "B1"], ["c1"],
                         kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c1"], ["r1"]),
        helper.make_node("Conv", ["r1", "W2", "B2"], ["c2"],
                         kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c2"], ["r2"]),
        helper.make_node("GlobalAveragePool", ["r2"], ["g"]),
        helper.make_node("Flatten", ["g"], ["f"], axis=1),
        helper.make_node("MatMul", ["f", "W3"], ["mm"]),
        helper.make_node("Add", ["mm", "B3"], ["output"]),
    ]

    graph = helper.make_graph(
        nodes, "convnet",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, n, n])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])],
        inits)
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    return m


def flip_gather():
    """Channel flip via Gather(axis=1, indices=[3,2,1,0]).

    This is the substitution the lecture used to work around a compiler
    limitation, and on cv181x it compiles and silently does nothing. That is
    the point of the exercise.
    """
    idx = numpy_helper.from_array(np.array([3, 2, 1, 0], np.int64), "idx")
    nodes = [helper.make_node("Gather", ["input", "idx"], ["output"], axis=1)]
    graph = helper.make_graph(
        nodes, "flip_gather",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4, 2, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 2, 2])],
        [idx])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    return m


def dynamic_range():
    """Range(0, sum(input), 1) -- an output whose length depends on the data."""
    nodes = [
        helper.make_node("ReduceSum", ["input"], ["s"], keepdims=0),
        helper.make_node("Cast", ["s"], ["sf"], to=TensorProto.FLOAT),
        helper.make_node("Range", ["zero", "sf", "one"], ["output"]),
    ]
    inits = [numpy_helper.from_array(np.array(0.0, np.float32), "zero"),
             numpy_helper.from_array(np.array(1.0, np.float32), "one")]
    graph = helper.make_graph(
        nodes, "dynamic_range",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 8])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None])],
        inits)
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    return m


def save(model, path):
    onnx.checker.check_model(model)
    onnx.save(model, path)
    print(f"  {path}")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "onnx"
    os.makedirs(out, exist_ok=True)
    for n in (32, 64, 128, 256):
        save(convnet(n), f"{out}/convnet_{n}.onnx")
    save(flip_gather(), f"{out}/flip_gather.onnx")
    save(dynamic_range(), f"{out}/dynamic_range.onnx")


main()
