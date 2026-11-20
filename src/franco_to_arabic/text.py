"""Text normalization, segmentation and line alignment.

The same functions are used for training data, dev/holdout evaluation and
inference on the hidden test set, so the model always sees identical input.
"""

from __future__ import annotations

import re
import unicodedata

ARABIC_RE = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
LATIN_RE = re.compile(r"[A-Za-z]")
# Invisible formatting characters (zero-width, bidi marks, BOM). They carry no
# transliteration information and the model cannot predict them.
INVISIBLE_RE = re.compile(r"[​-‏-⁦-⁩﻿]")
APOSTROPHES_RE = re.compile(r"[‘’‛ʼʻ´`′]")
DASHES_RE = re.compile(r"[‐-―−]")
SPACES_RE = re.compile(r"[ \t  -   　]+")
# Sentence boundary: end punctuation (Latin or Arabic) followed by whitespace.
SENT_SPLIT_RE = re.compile(r"(?<=[.!?؟])\s+")


def _common(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = INVISIBLE_RE.sub("", text)
    text = SPACES_RE.sub(" ", text)
    # strip spaces at line edges but keep the line structure (blank lines too)
    return "\n".join(line.strip(" ") for line in text.split("\n"))


def normalize_franco(text: str) -> str:
    """Normalize Arabizi input.

    - NFKC (full-width / compatibility forms -> canonical, e.g. '²' -> '2')
    - unify apostrophe variants to "'" (used in 3', 7', 6' for غ خ ظ)
    - unify dash variants to '-'
    - collapse horizontal whitespace, strip line edges, drop invisible chars
    Case and ALL digits are preserved: digits are letters in Arabizi
    (2=ء/ق, 3=ع, 5=خ, 6=ط, 7=ح, 8=غ, 9=ق/ص) and case marks names /
    English words that are copied as-is into the target.
    """
    text = unicodedata.normalize("NFKC", text)
    text = APOSTROPHES_RE.sub("'", text)
    text = DASHES_RE.sub("-", text)
    return _common(text)


def clean_target(text: str) -> str:
    """Minimal target cleaning. Do NOT normalize letters (alef/hamza, ة/ه, ى/ي):
    CER is scored against the raw reference, so the model must reproduce its
    spelling conventions exactly. Only whitespace / invisible chars are touched."""
    return _common(text)


def arabic_ratio(text: str) -> float:
    return len(ARABIC_RE.findall(text)) / max(1, len(text))


def nonempty_lines(text: str) -> list[str]:
    return [line for line in text.split("\n") if line.strip()]


def split_sentences(line: str) -> list[str]:
    return [s for s in SENT_SPLIT_RE.split(line) if s.strip()]


def ratio_ok(src: str, tgt: str, lo: float = 0.6, hi: float = 1.15,
             short: int = 20, short_lo: float = 0.3, short_hi: float = 2.0) -> bool:
    """Length-ratio sanity check len(tgt)/len(src). The Egyptian side is ~0.82x
    the Franco side; very short segments get looser bounds."""
    r = len(tgt) / max(1, len(src))
    if len(src) < short:
        return short_lo <= r <= short_hi
    return lo <= r <= hi


def prefix_align(fl: list[str], el: list[str]) -> list[tuple[str, str]]:
    """Fallback for documents whose line counts differ.

    The usual cause (EDA, 01_eda) is a Franco side truncated at ~5k chars
    while the Egyptian side continues. Lines stay parallel from the start, so
    we zip from the top, stop at the first implausible length ratio, and drop
    the last pair when Franco ran out (its final line is likely cut mid-way).
    """
    pairs = []
    for s, t in zip(fl, el):
        if not ratio_ok(s, t):
            break
        pairs.append((s, t))
    if pairs and len(pairs) == len(fl) < len(el):
        pairs.pop()
    return pairs


def align_document(franco: str, egyptian: str, long_line: int = 300, prefix_fallback: bool = False):
    """Align a (normalized) document pair into segment pairs.

    Returns None when the non-empty line counts differ and `prefix_fallback`
    is off (or recovers nothing); otherwise a list of dicts with keys:
    line_idx, seg_idx, src, tgt, level ('line' / 'sentence' / '*_prefix').
    Lines longer than `long_line` chars are further split into sentences
    when both sides have the same number of sentences; otherwise the whole
    line is kept as one segment.
    """
    fl, el = nonempty_lines(franco), nonempty_lines(egyptian)
    if not fl or not el:
        return None
    suffix = ""
    if len(fl) == len(el):
        pairs = list(zip(fl, el))
    elif prefix_fallback:
        pairs, suffix = prefix_align(fl, el), "_prefix"
        if not pairs:
            return None
    else:
        return None
    out = []
    for i, (s, t) in enumerate(pairs):
        if len(s) > long_line:
            ss, ts = split_sentences(s), split_sentences(t)
            if len(ss) == len(ts) and len(ss) > 1:
                for j, (a, b) in enumerate(zip(ss, ts)):
                    out.append(dict(line_idx=i, seg_idx=j, src=a, tgt=b, level="sentence" + suffix))
                continue
        out.append(dict(line_idx=i, seg_idx=0, src=s, tgt=t, level="line" + suffix))
    return out