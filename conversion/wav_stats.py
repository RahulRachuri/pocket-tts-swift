import sys, numpy as np
d = np.load(sys.argv[1]); w = d["wav"]
print(f"samples={w.size} peak={np.abs(w).max():.4f} rms={np.sqrt((w**2).mean()):.4f} "
      f"nonzero={(np.abs(w)>1e-4).mean()*100:.1f}% dtype={w.dtype}")
for k in ("step/latent","step/cond","step/eos_logit","mimi/out"):
    v = d[k]; print(f"{k:18s} shape={v.shape} min={v.min():.4f} max={v.max():.4f} "
                    f"finite={np.isfinite(v).all()}")
