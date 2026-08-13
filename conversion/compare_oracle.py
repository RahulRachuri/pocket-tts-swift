"""Bit-compare two oracle npz captures, key by key."""
import sys
import numpy as np

a = np.load(sys.argv[1]); b = np.load(sys.argv[2])
ka, kb = set(a.files), set(b.files)
if ka != kb:
    print("KEY MISMATCH:", sorted(ka ^ kb))
bad = 0
for k in sorted(ka & kb):
    x, y = a[k], b[k]
    if x.shape != y.shape:
        print(f"  SHAPE  {k}: {x.shape} vs {y.shape}"); bad += 1; continue
    if np.array_equal(x, y):
        continue
    d = np.abs(x.astype(np.float64) - y.astype(np.float64))
    print(f"  DIFF   {k:36s} max|d|={d.max():.3e} mean|d|={d.mean():.3e} ndiff={(d>0).sum()}/{d.size}")
    bad += 1
print(("BIT-IDENTICAL on all %d keys" % len(ka)) if bad == 0 else f"{bad} keys differ")
