#!/usr/bin/env python3
"""Ask the accelerator for memory until it kills us.

    sudo python3 ion_exhaust.py [chunk_mb]

The whole Cvitek stack allocates from an ION carveout.  This asks for the
carveout in chunks until the vendor library cannot satisfy the request, and
then keeps going, so you can see what the library does about it.

Nothing here is submitted to the TPU.  It only allocates.
"""
import ctypes
import sys
import time

LIBDIR = "/mnt/system/lib"
USED = "/sys/kernel/debug/ion/cvi_carveout_heap_dump/used_mem"


def load():
    ctypes.CDLL(f"{LIBDIR}/libcvikernel.so", mode=ctypes.RTLD_GLOBAL)
    lib = ctypes.CDLL(f"{LIBDIR}/libcviruntime.so", mode=ctypes.RTLD_GLOBAL)
    lib.CVI_RT_Init.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.CVI_RT_Init.restype = ctypes.c_int
    lib.CVI_RT_MemAlloc.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    lib.CVI_RT_MemAlloc.restype = ctypes.c_void_p
    lib.CVI_RT_MemGetPAddr.argtypes = [ctypes.c_void_p]
    lib.CVI_RT_MemGetPAddr.restype = ctypes.c_uint64
    return lib


def used_mb():
    try:
        return int(open(USED).read().strip()) / (1 << 20)
    except OSError:
        return float("nan")


def main():
    chunk = (int(sys.argv[1]) if len(sys.argv) > 1 else 8) << 20
    lib = load()
    rt = ctypes.c_void_p()
    print(f"CVI_RT_Init -> rc={lib.CVI_RT_Init(ctypes.byref(rt))}")

    held = []
    for i in range(1, 65):
        print(f"[{i:2d}] used={used_mb():6.1f} MB  CVI_RT_MemAlloc({chunk >> 20} MB) ...",
              end="", flush=True)
        mem = lib.CVI_RT_MemAlloc(rt, chunk)
        if not mem:
            print(" NULL")
            break
        held.append(mem)
        print(f" paddr=0x{lib.CVI_RT_MemGetPAddr(mem):x}")
        time.sleep(0.05)

    print(f"held {len(held)} chunks, used={used_mb():.1f} MB")
    print("keeping them allocated; exiting now frees them")


main()
