#!/usr/bin/env python3
"""Fit t(N) = F + k * N^2 to a few measurements.

    python3 fit.py 32 64 128 256 0.28 0.31 0.55 1.62

Times in milliseconds.  Least squares on two unknowns, so it is a 2x2 solve.

The convnet in models/ does 216*N^2 multiply-adds in the first convolution and
576*N^2 in the second (8 outputs x 3 inputs x 3x3, then 8 x 8 x 3x3), so 792*N^2
MACs, and 1584*N^2 FLOPs counting a MAC as two operations.  The global average
pool and the final matmul are 10*8 + 80, which is nothing.
"""
import sys

MACS_PER_N2 = 792.0


def solve(ns, ts):
    # design matrix rows are [1, n^2]
    s11 = float(len(ns))
    s12 = sum(n * n for n in ns)
    s22 = sum(n ** 4 for n in ns)
    b1 = sum(ts)
    b2 = sum(t * n * n for n, t in zip(ns, ts))
    det = s11 * s22 - s12 * s12
    if det == 0:
        raise SystemExit("need at least two different sizes")
    F = (b1 * s22 - b2 * s12) / det
    k = (s11 * b2 - s12 * b1) / det
    return F, k


def main():
    if len(sys.argv) < 5:
        print(__doc__)
        return 1
    half = (len(sys.argv) - 1) // 2
    ns = [float(v) for v in sys.argv[1:1 + half]]
    ts = [float(v) for v in sys.argv[1 + half:]]
    if len(ns) != len(ts):
        print(__doc__)
        return 1

    F, k = solve(ns, ts)
    print(f"F = {F:8.3f} ms   (fixed cost per inference)")
    print(f"k = {k:10.6f} ms per N^2\n")
    print(f"{'N':>6} {'measured':>10} {'model':>10} {'resid':>8} "
          f"{'MFLOP':>8} {'GOPS':>8} {'GOPS net of F':>14}")
    for n, t in zip(ns, ts):
        pred = F + k * n * n
        mflop = 2.0 * MACS_PER_N2 * n * n / 1e6
        gops = mflop / t
        net = mflop / (t - F) if t > F else float("nan")
        print(f"{n:6.0f} {t:10.3f} {pred:10.3f} {t - pred:+8.3f} "
              f"{mflop:8.3f} {gops:8.2f} {net:14.2f}")
    print("\npeak is 500 GOPS (0.5 TOPS).  Compare the last column.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
