"""Localize the cpu_only divergence: compare every graph output, frame 0, cpu vs gpu vs eager."""
import asyncio, sys
import numpy as np, torch
sys.path.insert(0, "conversion")
from export_mimi_decoder import MimiDecoderStep, STATE_NAMES, OUT_NAMES, init_state, RING
import coreai.runtime as rt
from pocket_tts import TTSModel

torch.set_num_threads(1)
orc = np.load("oracle/orc_a.npz")
lat = torch.from_numpy(orc["mimi/in"]).unsqueeze(-1)
step = MimiDecoderStep(TTSModel.load_model().eval().mimi).eval()
st = init_state(torch.float32)
with torch.no_grad():
    ref = step(lat[0:1], *[st[k] for k in STATE_NAMES])

async def run(unit):
    opts = (rt.SpecializationOptions.cpu_only() if unit == "cpu"
            else rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu()))
    m = await rt.AIModel.load("artifacts/mimi_decoder_float32_ring272.aimodel", opts)
    fn = m.load_function("main")
    feed = {"latent": rt.NDArray(lat[0:1].numpy())}
    feed.update({k: rt.NDArray(v.numpy()) for k, v in st.items()})
    r = await fn(feed)
    print(f"--- {unit} (frame 0) ---")
    for i, name in enumerate(OUT_NAMES):
        g = np.asarray(r[name].numpy()); e = ref[i].numpy()
        d = np.abs(g.astype(np.float64) - e.astype(np.float64))
        print(f"  {name:12s} {str(g.shape):20s} max|Δ|={d.max():.4e}  "
              f"ref[rms]={np.sqrt((e.astype(np.float64)**2).mean()):.4e} "
              f"got[rms]={np.sqrt((g.astype(np.float64)**2).mean()):.4e}")

for u in ("gpu", "cpu"):
    asyncio.run(run(u))
