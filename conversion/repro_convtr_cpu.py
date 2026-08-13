"""Minimal repro: ConvTranspose1d is WRONG on the Core AI cpu_only delegate.

Env: coreai-torch 0.4.1, coreai-core 1.0.0b2, coreai-opt 0.2.1, torch 2.11.0,
macOS 27 beta, Xcode 27 beta. GPU delegate is correct to ~1e-7 on every case below;
cpu_only returns uncorrelated values for every ConvTranspose1d case, while the plain
Conv1d control is exact on both. Found while porting the pocket-tts Mimi decoder,
whose first op is mimi.upsample.convtr = ConvTranspose1d(512,512,k=32,s=16,groups=512).
"""
import asyncio, shutil
from pathlib import Path
import numpy as np, torch, torch.nn as nn
import coreai.runtime as rt
from coreai_models.export.macos import export_to_coreai

torch.manual_seed(0)


class Wrap(nn.Module):
    def __init__(self, c):
        super().__init__(); self.c = c
    def forward(self, x):
        return self.c(x)


async def gate(path, x, ref, unit):
    opts = (rt.SpecializationOptions.cpu_only() if unit == "cpu"
            else rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu()))
    m = await rt.AIModel.load(str(path), opts)
    r = await m.load_function("main")({"x": rt.NDArray(x.numpy())})
    y = torch.from_numpy(np.asarray(r["y"].numpy()))
    cos = float(torch.nn.functional.cosine_similarity(y.reshape(1, -1), ref.reshape(1, -1)))
    print(f"    {unit:3s} max|Δ|={float((y - ref).abs().max()):.3e}  cos={cos:.6f}"
          f"  {'OK' if cos > 0.9999 else 'WRONG'}")


CASES = [
    ("CT1d C=64 k=4  s=2  T=4", nn.ConvTranspose1d(64, 64, 4, 2, bias=False), (1, 64, 4)),
    ("CT1d C=64 k=6  s=3  T=4", nn.ConvTranspose1d(64, 64, 6, 3, bias=False), (1, 64, 4)),
    ("CT1d C=64 k=8  s=4  T=4", nn.ConvTranspose1d(64, 64, 8, 4, bias=False), (1, 64, 4)),
    ("CT1d C=64 k=10 s=5  T=4", nn.ConvTranspose1d(64, 64, 10, 5, bias=False), (1, 64, 4)),
    ("CT1d C=64 k=12 s=6  T=4", nn.ConvTranspose1d(64, 64, 12, 6, bias=False), (1, 64, 4)),
    ("CT1d C=64 k=16 s=8  T=4", nn.ConvTranspose1d(64, 64, 16, 8, bias=False), (1, 64, 4)),
    ("CT1d C=64 k=20 s=10 T=4", nn.ConvTranspose1d(64, 64, 20, 10, bias=False), (1, 64, 4)),
    ("CT1d C=64 k=24 s=12 T=4", nn.ConvTranspose1d(64, 64, 24, 12, bias=False), (1, 64, 4)),
    ("CT1d C=64 k=32 s=16 T=4", nn.ConvTranspose1d(64, 64, 32, 16, bias=False), (1, 64, 4)),
]

for name, c, shape in CASES:
    mod = Wrap(c).eval()
    x = torch.randn(*shape)
    with torch.no_grad():
        ref = mod(x)
    prog = export_to_coreai(mod, {"x": torch.zeros(*shape)}, dynamic_shapes=None,
                            input_names=("x",), output_names=("y",),
                            state_names=None, externalize_modules=[])
    prog.optimize()
    p = Path("artifacts") / f"_repro_{abs(hash(name))}.aimodel"
    shutil.rmtree(p, ignore_errors=True)
    prog.save_asset(p, rt.AIModelAssetMetadata())
    print(f"  {name}")
    for u in ("gpu", "cpu"):
        asyncio.run(gate(p, x, ref, u))
    shutil.rmtree(p, ignore_errors=True)
