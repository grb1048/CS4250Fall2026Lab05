#!/usr/bin/env python3
"""Compare two WAV files: normalised cross-correlation and RMS.

    python3 corr.py a.wav b.wav

Standard library only, because there is no numpy on the board.  The two files
are 66,048 samples each, so a brute-force correlation is fine.
"""
import array
import math
import sys
import wave


def read(path):
    with wave.open(path, "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise SystemExit(f"{path}: expected mono 16-bit, got "
                             f"{w.getnchannels()}ch {w.getsampwidth() * 8}bit")
        a = array.array("h")
        a.frombytes(w.readframes(w.getnframes()))
        return a, w.getframerate()


def rms(x):
    return math.sqrt(sum(v * v for v in x) / len(x))


def corr(x, y, lag):
    """Normalised correlation of x with y shifted by lag samples."""
    if lag >= 0:
        a, b = x[:len(x) - lag], y[lag:]
    else:
        a, b = x[-lag:], y[:len(y) + lag]
    n = min(len(a), len(b))
    sa = sum(a[i] * b[i] for i in range(n))
    aa = sum(a[i] * a[i] for i in range(n))
    bb = sum(b[i] * b[i] for i in range(n))
    if aa == 0 or bb == 0:
        return 0.0
    return sa / math.sqrt(aa * bb)


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    x, fx = read(sys.argv[1])
    y, fy = read(sys.argv[2])
    print(f"{sys.argv[1]:20s} {len(x):6d} samples  {fx} Hz  {len(x)/fx:.3f} s  rms={rms(x):8.1f}")
    print(f"{sys.argv[2]:20s} {len(y):6d} samples  {fy} Hz  {len(y)/fy:.3f} s  rms={rms(y):8.1f}")
    if len(x) != len(y):
        print("lengths differ; comparing the overlap only")

    best = max(range(-64, 65), key=lambda lag: corr(x, y, lag))
    print(f"correlation at lag 0   : {corr(x, y, 0):+.4f}")
    print(f"best lag {best:+4d} samples : {corr(x, y, best):+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
