#!/usr/bin/env python3
"""Run a .cvimodel on the TPU through the bare CVI_NN API.

    sudo python3 npurun.py <model.cvimodel> [iterations]

The Cvitek runtime is a shared library with no Python bindings, so we call it
through ctypes and declare the structs by hand from cviruntime.h. There is no
compiler on the board, which turns out to be a feature: this is a text file you
can edit over ssh.

Everything except the five calls is written for you. Fill in the TODOs.
"""
import ctypes
import sys
import time

LIBDIR = "/mnt/system/lib"


class CVI_SHAPE(ctypes.Structure):
    _fields_ = [("dim", ctypes.c_int32 * 6), ("dim_size", ctypes.c_size_t)]


class CVI_TENSOR(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("shape", CVI_SHAPE),
        ("fmt", ctypes.c_int),
        ("count", ctypes.c_size_t),
        ("mem_size", ctypes.c_size_t),
        ("sys_mem", ctypes.c_void_p),
        ("paddr", ctypes.c_uint64),
        ("mem_type", ctypes.c_int),
        ("qscale", ctypes.c_float),
        ("zero_point", ctypes.c_int),
        ("pixel_format", ctypes.c_int),
        ("aligned", ctypes.c_bool),
        ("mean", ctypes.c_float * 3),
        ("scale", ctypes.c_float * 3),
        ("owner", ctypes.c_void_p),
        ("reserved", ctypes.c_char * 32),
    ]


TP = ctypes.POINTER(CVI_TENSOR)
FMT = {0: "fp32", 1: "int32", 2: "uint32", 3: "bf16",
       4: "int16", 5: "uint16", 6: "int8", 7: "uint8"}


def load():
    """Load the runtime and declare every signature we use."""
    ctypes.CDLL(f"{LIBDIR}/libcvikernel.so", mode=ctypes.RTLD_GLOBAL)
    lib = ctypes.CDLL(f"{LIBDIR}/libcviruntime.so", mode=ctypes.RTLD_GLOBAL)

    lib.CVI_NN_RegisterModel.argtypes = [ctypes.c_char_p,
                                        ctypes.POINTER(ctypes.c_void_p)]
    lib.CVI_NN_RegisterModel.restype = ctypes.c_int

    lib.CVI_NN_GetInputOutputTensors.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(TP), ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(TP), ctypes.POINTER(ctypes.c_int32)]
    lib.CVI_NN_GetInputOutputTensors.restype = ctypes.c_int

    lib.CVI_NN_Forward.argtypes = [ctypes.c_void_p, TP, ctypes.c_int32,
                                   TP, ctypes.c_int32]
    lib.CVI_NN_Forward.restype = ctypes.c_int

    lib.CVI_NN_CleanupModel.argtypes = [ctypes.c_void_p]
    lib.CVI_NN_CleanupModel.restype = ctypes.c_int

    lib.CVI_NN_GetModelVersion.argtypes = [ctypes.c_void_p,
                                           ctypes.POINTER(ctypes.c_int32),
                                           ctypes.POINTER(ctypes.c_int32)]
    lib.CVI_NN_GetModelVersion.restype = ctypes.c_int

    lib.CVI_NN_GetModelTarget.argtypes = [ctypes.c_void_p]
    lib.CVI_NN_GetModelTarget.restype = ctypes.c_char_p
    return lib


def describe(t):
    dims = list(t.shape.dim[:t.shape.dim_size])
    return (f"{(t.name or b'?').decode():24s} dims={dims} "
            f"fmt={FMT.get(t.fmt, t.fmt)} bytes={t.mem_size} "
            f"qscale={t.qscale:.6f}")


# ---------------------------------------------------------------------------
# The five calls.  Each returns what the API returns: a 0 for success.
# ---------------------------------------------------------------------------

def register(lib, path):
    """Load a model.  Returns a handle (c_void_p) or None.

    ctypes: the model handle is an opaque c_void_p that the library fills in,
    so you pass ctypes.byref(model) and check the returned rc.
    """
    # TODO
    raise NotImplementedError


def get_tensors(lib, model):
    """Ask the model what its inputs and outputs look like.

    Returns (ins, n_in, outs, n_out) where ins/outs are POINTER(CVI_TENSOR)
    arrays of n_in / n_out entries.  You will need four locals: two TP() and
    two ctypes.c_int32(), all passed by reference.
    """
    # TODO
    raise NotImplementedError


def fill_input(t, value=128):
    """Fill an input tensor with a constant.

    The buffer is already allocated by the runtime; t.sys_mem points at it and
    t.mem_size is its length in bytes.
    """
    # TODO
    raise NotImplementedError


def forward(lib, model, ins, n_in, outs, n_out):
    """Run the model once."""
    # TODO
    raise NotImplementedError


def cleanup(lib, model):
    """Release the model.  This frees its ION allocations."""
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    path = sys.argv[1]
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    lib = load()
    model = register(lib, path)
    if model is None:
        return 1

    try:
        maj, mnr = ctypes.c_int32(), ctypes.c_int32()
        if lib.CVI_NN_GetModelVersion(model, ctypes.byref(maj),
                                      ctypes.byref(mnr)) == 0:
            print(f"model version {maj.value}.{mnr.value}")
        print(f"target: {lib.CVI_NN_GetModelTarget(model)}")

        ins, n_in, outs, n_out = get_tensors(lib, model)
        for i in range(n_in.value):
            print(f"  in[{i}]  {describe(ins[i])}")
            fill_input(ins[i])
        for i in range(n_out.value):
            print(f"  out[{i}] {describe(outs[i])}")

        t0 = time.perf_counter()
        rc = forward(lib, model, ins, n_in, outs, n_out)
        first = (time.perf_counter() - t0) * 1000
        print(f"first forward -> rc={rc} in {first:.2f} ms")

        t0 = time.perf_counter()
        for _ in range(iters):
            rc = forward(lib, model, ins, n_in, outs, n_out)
            if rc != 0:
                print(f"CVI_NN_Forward -> rc={rc}")
                return 1
        dt = (time.perf_counter() - t0) / iters
        print(f"{iters} iterations: {dt * 1000:.3f} ms/inference "
              f"({1.0 / dt:.1f} FPS)")

        for i in range(min(n_out.value, 2)):
            t = outs[i]
            buf = (ctypes.c_uint8 * t.mem_size).from_address(t.sys_mem)
            print(f"  out[{i}] first 16 bytes: {list(buf[:16])}")
        return 0
    finally:
        cleanup(lib, model)


if __name__ == "__main__":
    sys.exit(main())
