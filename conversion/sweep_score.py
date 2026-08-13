"""Score the sweep: ASR round-trip WER + audio sanity, then aggregate.

Gate 1 — intelligibility. Every clip is resampled 24 kHz float32 -> 16 kHz mono PCM16
and transcribed by the local `parakeet-swift` host, driven through its `serve` mode
(one process, one JSON request per line) so ~300 clips cost one model load instead of
300. WER is jiwer's, on a normalizer that removes the two things that make a TTS/ASR
round-trip lie: casing/punctuation, and the digit-vs-word split (`1851` on one side,
`eighteen fifty one` on the other). Numbers are spelled out on BOTH sides before the
comparison, so "$1,250" -> "one thousand two hundred and fifty dollars"; the TTS may
still legitimately say "twelve fifty", which is why numeric rows carry a `num` flag and
are reported as their own bucket rather than folded into the headline.

Gate 2 — sanity, which catches the failures WER cannot see because a truncated clip can
still transcribe cleanly:
    empty          no audio at all
    low_level      RMS far under THIS VOICE's median (the eight shipped voices sit
                   ~12 dB apart, so a fixed threshold just re-flags the quiet ones)
    short_audio    duration far below what the word count implies
    long_audio     runaway generation
    clipping       peak >= 0.99
    no_eos         a chunk that never fired EOS
    kv_capped      a chunk that ran into the static KV capacity (S_MAX)
    chunk_over_max upstream's chunker emitted a chunk over MAX_TOKEN_PER_CHUNK

    python conversion/sweep_score.py --meta artifacts/sweep_fp32_gen.jsonl \
        --wav-dir artifacts/sweep_fp32 --out artifacts/sweep_fp32_scored.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import scipy.signal

# External ASR dependency; see conversion/asr_gate.py for the same two overrides.
PARAKEET = Path(os.environ.get("PARAKEET_BIN", "parakeet-swift"))
PK_ARTIFACTS = Path(os.environ.get("PARAKEET_ARTIFACTS", "artifacts_v2"))

# words a TTS/ASR round-trip renders inconsistently but that carry no information
_ORDINAL = {"st": "", "nd": "", "rd": "", "th": ""}


def _spell(tok: str) -> str:
    from num2words import num2words

    t = tok.replace(",", "")
    m = re.fullmatch(r"\$?(\d+(?:\.\d+)?)(st|nd|rd|th)?", t)
    if not m:
        return tok
    val, ord_suf = m.group(1), m.group(2)
    try:
        if ord_suf:
            out = num2words(int(float(val)), to="ordinal")
        elif "." in val:
            out = num2words(float(val))
        else:
            n = int(val)
            # A bare 4-digit year is read as a PAIR by every narrator ("eighteen fifty
            # one"), never as a cardinal, so spell it that way — otherwise a correct
            # transcription scores five substitutions against "one thousand eight
            # hundred fifty one" and the numeric bucket measures the normalizer instead
            # of the model.
            if 1100 <= n <= 2099 and not tok.startswith("$"):
                hi, lo = divmod(n, 100)
                out = (
                    f"{num2words(hi)} hundred"
                    if lo == 0
                    else f"{num2words(hi)} oh {num2words(lo)}"
                    if lo < 10
                    else f"{num2words(hi)} {num2words(lo)}"
                )
            else:
                out = num2words(n)
        if tok.startswith("$"):
            out += " dollars"
        # num2words writes "one hundred and one"; spoken English drops the "and" about
        # as often as it keeps it, so it is removed on both sides rather than gambled on
        return out.replace(" and ", " ")
    except Exception:
        return tok


def normalize(s: str) -> str:
    s = s.lower()
    s = s.replace("&", " and ").replace("%", " percent ").replace("+", " plus ")
    s = re.sub(r"[‘’]", "'", s)
    s = re.sub(r"[^a-z0-9$.,' ]+", " ", s)
    s = re.sub(r"(?<=[a-z])\.(?=[a-z])", "", s)  # u.s.a -> usa, a.m -> am
    toks = []
    for t in s.split():
        t = t.strip(".,'")
        if not t:
            continue
        if re.search(r"\d", t):
            t = _spell(t)
        toks.append(t)
    s = " ".join(toks)
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    # hyphenated / joined number words differ across writers: "twenty-one" vs "twenty one"
    s = s.replace("-", " ")
    return re.sub(r"\s+", " ", s).strip()


def wer_of(ref: str, hyp: str) -> tuple[float, int, int]:
    import jiwer

    r, h = normalize(ref), normalize(hyp)
    if not r:
        return 0.0, 0, 0
    o = jiwer.process_words(r, h if h else "")
    n = o.substitutions + o.deletions + o.hits
    return o.wer, int(o.substitutions + o.deletions + o.insertions), int(n)


def load_16k(path: Path) -> np.ndarray:
    sr, x = scipy.io.wavfile.read(path)
    x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.abs(x).max() > 1.5:
        x = x / 32768.0
    if sr != 16000:
        g = np.gcd(sr, 16000)
        x = scipy.signal.resample_poly(x, 16000 // g, sr // g)
    return x


def to_16k(path: Path, out: Path) -> None:
    """Default framing: 0.25 s of silence either side, floor of 2 s."""
    x = load_16k(path)
    pad = int(0.25 * 16000)
    x = np.concatenate([np.zeros(pad, np.float32), x, np.zeros(pad, np.float32)])
    if x.size < 32000:
        x = np.pad(x, (0, 32000 - x.size))
    scipy.io.wavfile.write(out, 16000, (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16))


def _write(x: np.ndarray, out: Path) -> None:
    scipy.io.wavfile.write(out, 16000, (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16))


def _frame(x: np.ndarray, mode: str) -> tuple[np.ndarray, int]:
    """Returns (audio, repeat_factor). `repeat` is how many copies of the reference the
    transcript should be scored against."""
    z = lambda s: np.zeros(int(s * 16000), np.float32)  # noqa: E731
    if mode == "pad025":
        y = np.concatenate([z(0.25), x, z(0.25)])
        return (np.pad(y, (0, max(0, 32000 - y.size))), 1)
    if mode == "pad1":
        return np.concatenate([z(1.0), x, z(1.0)]), 1
    if mode == "x3":
        return np.concatenate([z(0.5), x, z(0.5), x, z(0.5), x, z(0.5)]), 3
    raise ValueError(mode)


def _serve(paths: list[str]) -> list[str]:
    """One `parakeet-swift serve` process for a whole batch: model loaded once."""
    if not paths:
        return []
    env = dict(os.environ, PARAKEET_ARTIFACTS=str(PK_ARTIFACTS))
    p = subprocess.Popen(
        [str(PARAKEET), "serve"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        env=env,
    )
    stdout, _ = p.communicate("".join(json.dumps({"audio": a}) + "\n" for a in paths))
    lines = [l for l in stdout.splitlines() if l.strip().startswith("{")]
    out = []
    for i in range(len(paths)):
        try:
            out.append(json.loads(lines[i]).get("text", ""))
        except Exception:
            out.append("")
    return out


def _looped(hyp: str, ref: str) -> bool:
    """A short n-gram repeated 3+ times back to back in the transcript but not in the
    reference. This is the small-scale form of the ASR runaway ("Indeed, Indeed Indeed
    Indeed", "Aushaks Aushaks Aushaks") — too few words to trip a length test, but the
    same decoder failure, and it lands entirely on 1-3 word clips."""
    h = normalize(hyp).split()
    r = normalize(ref).split()
    for k in (1, 2, 3):
        for i in range(len(h) - 3 * k + 1):
            g = h[i : i + k]
            if g * 3 == h[i : i + 3 * k] and " ".join(g * 2) not in " ".join(r):
                return True
    return False


def _implausible(hyp: str, dur_s: float) -> str | None:
    """The ASR host is framing-sensitive in both directions: it returns an EMPTY
    transcript for an isolated sub-2-second utterance no matter how much silence is
    added, and it can run away into a token loop (measured: 1445 words out of a 7.1 s
    clip whose audio is a clean, correct 10-word count) on one particular padding and
    not on another. Both are gate failures, not TTS failures, so they are detected here
    and retried under a different framing rather than charged to the model."""
    n = len(hyp.split())
    if n == 0:
        return "empty"
    if n > 3.0 * dur_s * WPS + 15:
        return "runaway"
    return None


def transcribe_robust(
    wavs: list[Path], refs: dict[str, str] | None = None
) -> dict[str, tuple[str, int, str]]:
    """stem -> (transcript, reference repeat factor, framing used / 'unreliable')."""
    refs = refs or {}
    tmp = Path(tempfile.mkdtemp())
    audio = {w.stem: load_16k(w) for w in wavs}
    dur = {k: v.size / 16000 for k, v in audio.items()}
    result: dict[str, tuple[str, int, str]] = {}

    pending = [w.stem for w in wavs]
    for attempt, mode in enumerate(("pad025", "x3", "pad1")):
        if not pending:
            break
        paths, reps = [], {}
        for s in pending:
            y, rep = _frame(audio[s], mode)
            p = tmp / f"{s}_{mode}.wav"
            _write(y, p)
            paths.append(str(p))
            reps[s] = rep
        hyps = _serve(paths)
        nxt = []
        for s, h in zip(pending, hyps):
            # score the retry framings against the audio they actually contain
            bad = _implausible(h, dur[s] * reps[s]) or (
                "loop" if _looped(h, " ".join([refs.get(s, "")] * reps[s])) else None
            )
            if bad is None:
                result[s] = (h, reps[s], mode)
            elif attempt == 2:
                result[s] = (h, reps[s], "unreliable")
            else:
                nxt.append(s)
        pending = nxt
        if pending:
            print(f"[asr]   {len(pending)} clip(s) implausible after {mode}, retrying", flush=True)
    return result


def transcribe_all(wavs: list[Path]) -> dict[str, str]:
    return {k: v[0] for k, v in transcribe_robust(wavs).items()}


WPS = 3.1  # median words/second measured across the 302-clip fp32 sweep


def flags_for(row: dict, base_rms: float = 0.09) -> list[str]:
    f = []
    dur, words = row.get("duration_s", 0.0), max(row.get("words", 1), 1)
    if row.get("error"):
        return ["error"]
    if row.get("samples", 0) == 0:
        f.append("empty")
    else:
        if dur < 0.55 * words / WPS:
            f.append("short_audio")
        if dur > 2.5 * words / WPS + 3.0:
            f.append("long_audio")
        if row.get("rms", 1.0) < 0.45 * base_rms:
            f.append("low_level")
    if row.get("peak", 0.0) >= 0.99:
        f.append("clipping")
    if row.get("no_eos"):
        f.append("no_eos")
    if row.get("kv_capped"):
        f.append("kv_capped")
    if any(c.get("tokens", 0) > 50 for c in row.get("chunks", [])):
        f.append("chunk_over_max")
    if any(c.get("hit_max_gen_len") for c in row.get("chunks", [])):
        f.append("hit_max_gen_len")
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.10)
    a = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    rows = [json.loads(l) for l in (root / a.meta).read_text().splitlines() if l.strip()]
    wdir = root / a.wav_dir
    wavs = [wdir / r["wav"] for r in rows if r.get("wav") and (wdir / r["wav"]).exists()]
    print(f"[asr] transcribing {len(wavs)} clips ...", flush=True)
    hyp = transcribe_robust(wavs, {Path(r["wav"]).stem: r["text"] for r in rows if r.get("wav")})

    import statistics

    byv = defaultdict(list)
    for r in rows:
        if r.get("rms"):
            byv[r.get("voice")].append(r["rms"])
    vmed = {k: statistics.median(v) for k, v in byv.items()}
    print("  per-voice median RMS:", {k: round(v, 4) for k, v in sorted(vmed.items())})

    scored = []
    for r in rows:
        h, rep, mode = hyp.get(Path(r.get("wav", "")).stem, ("", 1, "missing"))
        ref = " ".join([r["text"]] * rep)
        w, errs, n = wer_of(ref, h)
        # a x3 framing triples both sides; fold back so pooled counts stay per-clip
        errs, n = int(round(errs / rep)), int(round(n / rep))
        fl = flags_for(r, vmed.get(r.get("voice"), 0.09)) + (
            ["asr_unreliable"] if mode == "unreliable" else []
        )
        scored.append({**r, "hyp": h, "asr_framing": mode, "wer": round(float(w), 4),
                       "errs": errs, "ref_words": n, "flags": fl})
    outp = root / a.out
    outp.write_text("\n".join(json.dumps(s) for s in scored) + "\n")

    # rows whose ASR round trip is not trustworthy are excluded from every WER number
    # and reported on their own; charging them to the model would be a lie in both
    # directions (empty transcript -> 100%, token loop -> 1400%).
    usable = [s for s in scored if "asr_unreliable" not in s["flags"]]
    ok = [s for s in usable if not s["flags"]]
    tot_e = sum(s["errs"] for s in usable)
    tot_n = sum(s["ref_words"] for s in usable)
    print(f"\n[aggregate] {len(scored)} clips  ({len(usable)} ASR-usable, {len(ok)} unflagged)")
    print(f"  corpus WER (all usable)     {100 * tot_e / max(tot_n, 1):.2f}%")
    e2 = sum(s["errs"] for s in ok)
    n2 = sum(s["ref_words"] for s in ok)
    print(f"  corpus WER (unflagged only) {100 * e2 / max(n2, 1):.2f}%")
    fr = defaultdict(int)
    for s in scored:
        fr[s["asr_framing"]] += 1
    print(f"  ASR framing used: {dict(fr)}")

    def table(key, title):
        g = defaultdict(lambda: [0, 0, 0, 0])
        for s in usable:
            k = s.get(key, "?")
            g[k][0] += s["errs"]
            g[k][1] += s["ref_words"]
            g[k][2] += 1
            g[k][3] += 1 if s["flags"] else 0
        print(f"  --- {title}")
        for k in sorted(g, key=str):
            e, n, c, fl = g[k]
            print(f"      {str(k):<14} n={c:<4} WER {100 * e / max(n, 1):6.2f}%   flagged {fl}")

    table("voice", "by voice")
    table("source", "by corpus source")
    table("bucket", "by length bucket")

    bad = [s for s in scored if s["wer"] > a.threshold or s["flags"]]
    print(f"\n  {len(bad)} clips over WER {a.threshold:.0%} or flagged -> {outp}")
    fc = defaultdict(int)
    for s in scored:
        for f in s["flags"]:
            fc[f] += 1
    print("  flag counts:", dict(fc))


if __name__ == "__main__":
    sys.exit(main())
