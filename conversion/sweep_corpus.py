"""Build the validation-sweep corpus.

Three sources, so a failure can be attributed to the *kind* of text rather than to
one book's idiom:

  librispeech  read-speech transcripts (test-clean + test-other). Upper-cased and
               unpunctuated at source, so they are lower-cased and given a leading
               capital here; that is exactly what the ASR gate's reference should be
               and it keeps the TTS from reading them as acronyms.
  moby         Project Gutenberg #2701 narrative prose — long clauses, semicolons,
               em dashes, archaic spelling. This is the closest thing to long-form
               production text, which is what the port targets.
  hard         hand-built adversarial cases: numbers, years, money, abbreviations,
               acronyms, invented proper nouns, 1-3 word utterances, 40+ word
               sentences. These are where a TTS normally breaks, and where the ASR
               round-trip is least trustworthy (see `flags` in the scorer).

Deterministic: one seed, sorted inputs, no wall-clock or dict-order dependence.

    .venv-export/bin/python conversion/sweep_corpus.py --out artifacts/sweep_corpus.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path

# Corpus sources are external data, not part of this repo. Point CORPUS_ROOT at a
# directory holding `librispeech/LibriSpeech` (test-clean + test-other) and
# `work/moby_dick_gutenberg.txt` (Project Gutenberg #2701, plain text).
BENCH = Path(os.environ.get("CORPUS_ROOT", "corpus"))
LIBRI = BENCH / "librispeech/LibriSpeech"
MOBY = BENCH / "work/moby_dick_gutenberg.txt"

# The 8 voices the checkpoint ships as its base catalogue (`embeddings/` in the HF
# repo). Conditioning lengths differ per voice and that is load-bearing for the KV
# budget, so the sweep records the voice on every row.
VOICES = ["alba", "azelma", "cosette", "eponine", "fantine", "javert", "jean", "marius"]


def libri_sentences() -> list[str]:
    out = []
    for f in sorted(LIBRI.rglob("*.trans.txt")):
        for line in f.read_text().splitlines():
            _, _, text = line.partition(" ")
            text = text.strip().lower()
            if text:
                out.append(text[0].upper() + text[1:])
    return out


def moby_sentences() -> list[str]:
    raw = MOBY.read_text(encoding="utf-8", errors="replace")
    # body only: between the first chapter heading and the epilogue
    start = raw.find("CHAPTER 1. Loomings.")
    body = raw[start:] if start > 0 else raw
    body = re.sub(r"CHAPTER \d+\.[^\n]*\n", " ", body)
    body = re.sub(r"\s+", " ", body)
    # unicode punctuation the tokenizer has no reason to know about
    body = body.replace("’", "'").replace("‘", "'")
    body = body.replace("“", '"').replace("”", '"')
    body = body.replace("—", " - ").replace("–", "-")
    parts = re.split(r"(?<=[.!?])\s+", body)
    out = []
    for p in parts:
        p = p.strip().strip('"').strip()
        if not p or not p[0].isalpha():
            continue
        if len(p.split()) < 3:
            continue
        out.append(p)
    return out


HARD: list[tuple[str, str]] = [
    # --- numbers, years, money, measures
    ("num", "The ship was built in 1851 and sank in 1873."),
    ("num", "It cost $1,250 and weighed 47 kilograms."),
    ("num", "Chapter 12 begins on page 308."),
    ("num", "She ran 26.2 miles in 3 hours and 41 minutes."),
    ("num", "The account balance fell to negative 17 dollars."),
    ("num", "Add 3 and 4 to get 7."),
    ("num", "In 2026 the population reached 8.1 billion."),
    ("num", "Room 101 is on the 4th floor."),
    ("num", "The temperature dropped to minus 40 degrees overnight."),
    ("num", "He paid 99 cents for the first one and $19.99 for the second."),
    # --- abbreviations and titles
    ("abbr", "Dr. Watson met Mr. Holmes at 221B Baker St. on Tuesday."),
    ("abbr", "St. Mary's Hospital is on Elm Ave. near the Jr. college."),
    ("abbr", "Prof. Adams, Ph.D., wrote the foreword, etc."),
    ("abbr", "The Rev. Mapple climbed the pulpit at 6 a.m."),
    ("abbr", "Capt. Ahab vs. the whale, Vol. II, ch. 4."),
    ("abbr", "Please see Fig. 3 and Table 2 for details."),
    # --- acronyms and initialisms
    ("acro", "The CPU and the GPU both failed the ANE test."),
    ("acro", "NASA and the FBI issued a joint FAQ."),
    ("acro", "Send the PDF to the CEO via HTTPS."),
    ("acro", "WER is measured by ASR after TTS."),
    ("acro", "The BBC reported that the UN and the EU disagreed."),
    ("acro", "IEEE 802.11 is a Wi-Fi standard."),
    # --- invented fantasy proper nouns
    ("fantasy", "Kaelthorn Vraskyr rode from Ilmenwyth to the gates of Zharudan."),
    ("fantasy", "The Thaumaturge Ysolde Vennarion bound the Skarn to her will."),
    ("fantasy", "Beyond Quel'Doreth lies the drowned city of Nyxhavel."),
    ("fantasy", "Orithane and Belzuvath argued over the Sunder Stone."),
    ("fantasy", "Grimwald Thistlebottom of Underhollow refused the summons."),
    ("fantasy", "The Aeltherim call it Vaskirion; the Drenn call it the Long Dark."),
    ("fantasy", "Xiuhtecuhtli Anaxagorou spoke the ninth syllable."),
    # --- very short (1-3 words)
    ("short", "Yes."),
    ("short", "No."),
    ("short", "Stop!"),
    ("short", "Who is there?"),
    ("short", "Not yet."),
    ("short", "Absolutely not."),
    ("short", "Wait."),
    ("short", "Good morning."),
    ("short", "Why?"),
    ("short", "Call me Ishmael."),
    # --- very long (40+ words, single sentence)
    (
        "long",
        "When the fog finally lifted from the harbour that morning the crew could at "
        "last make out the shape of the breakwater, the cranes standing idle against "
        "a pale and colourless sky, and beyond them the low grey warehouses where the "
        "cargo they had carried for eleven weeks would be unloaded before nightfall.",
    ),
    (
        "long",
        "It is a curious thing that a man who has spent his whole life at sea, who has "
        "weathered more storms than he can count and buried more friends than he cares "
        "to remember, will still stand at the rail on a calm evening and look out over "
        "the water with the same uncomplicated wonder he felt as a boy of twelve.",
    ),
    (
        "long",
        "The committee, having considered the evidence submitted by the three "
        "departments, having heard testimony from eleven witnesses over the course of "
        "four days, and having reviewed the financial records covering the preceding "
        "seven years, concluded that no single individual could reasonably be held "
        "responsible for the failure.",
    ),
    (
        "long",
        "She explained, in the patient and slightly weary tone of someone who has "
        "explained the same thing many times before, that the machine did not think, "
        "did not understand, did not intend anything at all, but merely predicted the "
        "next most plausible symbol given everything it had already been shown.",
    ),
    (
        "long",
        "Down the long corridor, past the reading room with its green lamps and its "
        "smell of old paper, past the cabinets of pressed flowers collected by a "
        "botanist dead for a century and a half, and past the door that nobody had "
        "opened in living memory, the two of them walked without speaking a word.",
    ),
    # --- mixed / punctuation stress
    ("mixed", "\"Stop!\" he cried; but the boat, already half swamped, drifted on."),
    ("mixed", "Well - and this is the strange part - nobody objected."),
    ("mixed", "He asked: why now? Why here? Why us?"),
    ("mixed", "The file (version 2.1, dated Jan. 3rd) is attached."),
    ("mixed", "It's the crew's boat, not the captain's."),
    ("mixed", "Re-entering, he re-read the co-operative's pre-arranged agreement."),
    ("mixed", "One, two, three, four, five, six, seven, eight, nine, ten."),
    ("mixed", "A B C D E F G."),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/sweep_corpus.jsonl")
    ap.add_argument("--seed", type=int, default=20260813)
    ap.add_argument("--n-libri", type=int, default=150)
    ap.add_argument("--n-moby", type=int, default=100)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = []

    # librispeech: stratify by word count so short and long utterances both appear
    libri = libri_sentences()
    buckets: dict[str, list[str]] = {"s": [], "m": [], "l": [], "xl": []}
    for s in libri:
        n = len(s.split())
        b = "s" if n <= 6 else "m" if n <= 15 else "l" if n <= 30 else "xl"
        buckets[b].append(s)
    share = {"s": 0.20, "m": 0.30, "l": 0.30, "xl": 0.20}
    for b, frac in share.items():
        k = min(int(round(args.n_libri * frac)), len(buckets[b]))
        rows += [("librispeech", b, s) for s in rng.sample(sorted(buckets[b]), k)]

    moby = moby_sentences()
    mb: dict[str, list[str]] = {"s": [], "m": [], "l": [], "xl": []}
    for s in moby:
        n = len(s.split())
        b = "s" if n <= 6 else "m" if n <= 15 else "l" if n <= 30 else "xl"
        mb[b].append(s)
    for b, frac in share.items():
        k = min(int(round(args.n_moby * frac)), len(mb[b]))
        rows += [("moby", b, s) for s in rng.sample(sorted(mb[b]), k)]

    rows += [("hard", kind, text) for kind, text in HARD]

    rng.shuffle(rows)

    # cycle the 8 voices over the shuffled corpus: each voice gets ~1/8 of every
    # source and every length bucket, which is what makes the per-voice table
    # comparable at all.
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for i, (src, bucket, text) in enumerate(rows):
            f.write(
                json.dumps(
                    {
                        "id": f"s{i:04d}",
                        "source": src,
                        "bucket": bucket,
                        "voice": VOICES[i % len(VOICES)],
                        "words": len(text.split()),
                        "text": text,
                    }
                )
                + "\n"
            )
    print(f"[corpus] {len(rows)} rows -> {out}")
    from collections import Counter

    print("  by source:", dict(Counter(r[0] for r in rows)))
    print("  by bucket:", dict(Counter(r[1] for r in rows)))


if __name__ == "__main__":
    main()
