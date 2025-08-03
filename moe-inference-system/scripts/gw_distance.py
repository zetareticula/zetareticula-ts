#!/usr/bin/env python3
"""Compute Gromov–Wasserstein distance via POT (Python Optimal Transport).
Reads JSON from stdin with keys X, Y (list of list) and optional epsilon, maxIter, tol.
Prints a single float distance to stdout.
"""
import json, sys, math
try:
    import numpy as np
    import ot
except ImportError as e:
    print("nan", end="")
    sys.exit(1)

def main():
    payload = json.load(sys.stdin)
    X = np.array(payload["X"], dtype=np.float32)
    Y = np.array(payload["Y"], dtype=np.float32)
    n, m = len(X), len(Y)
    px = np.ones(n) / n
    py = np.ones(m) / m
    Cx = ot.dist(X, X, metric='euclidean')
    Cy = ot.dist(Y, Y, metric='euclidean')
    gw = ot.gromov.gromov_wasserstein2(Cx, Cy, px, py, 'square_loss')
    print(float(math.sqrt(gw)), end="")

if __name__ == "__main__":
    main()
