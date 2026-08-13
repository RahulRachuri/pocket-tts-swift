"""Enumerate model shapes and the streaming-state schemas (flowlm KV + mimi)."""
import os, torch
from pocket_tts import TTSModel
from pocket_tts.modules.stateful_module import init_states, StatefulModule

torch.set_num_threads(1)
m = TTSModel.load_model()
print("has_voice_cloning:", m.has_voice_cloning)
print("sample_rate:", m.sample_rate, "frame_rate:", m.config.mimi.frame_rate)
print("ldim:", m.flow_lm.ldim, "dim:", m.flow_lm.dim)
print("mimi hop_length:", m.mimi.encoder.hop_length, "encoder_frame_rate:", m.mimi.encoder_frame_rate)
print("frame_size:", m.mimi.frame_size)
print("temp:", m.temp, "lsd_steps:", m.lsd_decode_steps, "noise_clamp:", m.noise_clamp, "eos_thr:", m.eos_threshold)
print("frames_after_eos:", m.model_recommended_frames_after_eos, "pad_short:", m.pad_with_spaces_for_short_inputs)
tot = sum(p.numel() for p in m.parameters())
print("params total: %.2fM" % (tot/1e6))
for name, mod in [("flow_lm", m.flow_lm), ("mimi", m.mimi)]:
    n = sum(p.numel() for p in mod.parameters())
    print(f"  {name}: {n/1e6:.2f}M")
print("  flow_lm.transformer: %.2fM" % (sum(p.numel() for p in m.flow_lm.transformer.parameters())/1e6))
print("  flow_lm.flow_net:    %.2fM" % (sum(p.numel() for p in m.flow_lm.flow_net.parameters())/1e6))
print("  flow_lm.conditioner: %.2fM" % (sum(p.numel() for p in m.flow_lm.conditioner.parameters())/1e6))
print("  mimi.decoder:        %.2fM" % (sum(p.numel() for p in m.mimi.decoder.parameters())/1e6))
print("  mimi.decoder_tr:     %.2fM" % (sum(p.numel() for p in m.mimi.decoder_transformer.parameters())/1e6))
print("  mimi.upsample:       %.2fM" % (sum(p.numel() for p in m.mimi.upsample.parameters())/1e6))
print("  mimi.quantizer:      %.2fM" % (sum(p.numel() for p in m.mimi.quantizer.parameters())/1e6))

print("\n=== FLOWLM STATE (init_states, seq_len=512) ===")
s = init_states(m.flow_lm, 1, 512)
for k, v in s.items():
    print(" ", k, {kk: (tuple(vv.shape), str(vv.dtype)) for kk, vv in v.items()})

print("\n=== MIMI STATE (init_states, seq_len=160) ===")
ms = init_states(m.mimi, 1, 160)
print("count:", len(ms), "tensors:", sum(len(v) for v in ms.values()))
for k, v in ms.items():
    print(" ", k, {kk: (tuple(vv.shape), str(vv.dtype)) for kk, vv in v.items()})

print("\n=== MIMI DECODE-PATH STATE ONLY (modules touched by decode_from_latent) ===")
touched = []
for prefix in ("upsample", "decoder_transformer", "decoder"):
    sub = getattr(m.mimi, prefix)
    for mn, mod in sub.named_modules():
        if isinstance(mod, StatefulModule):
            touched.append(f"{prefix}.{mn}" if mn else prefix)
print("modules:", len(touched))
ntens = 0
for t in touched:
    st = ms[t]
    ntens += len(st)
    print(" ", t, {kk: (tuple(vv.shape), str(vv.dtype)) for kk, vv in st.items()})
print("decode-path tensors:", ntens)
