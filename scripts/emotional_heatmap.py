#!/usr/bin/env python3
"""
Post-transcription emotional heatmap generator.

Inputs:
  - Subtitles: .srt, .vtt (and .svt treated like .srt)
  - Documents: .txt, .md, .pdf, .rtf, .docx
  - Fallback: any other file is treated as plain text (utf-8 with replacement)

Outputs (to --out-dir):
  - <stem>.emotion_bins.csv
  - <stem>.emotion_heatmap.png
  - <stem>.emotion_bins.json

This script is intentionally offline-first: it uses a small built-in lexical
scoring model (keyword-based) so it runs without external APIs.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclasses.dataclass(frozen=True)
class Segment:
    start_sec: Optional[float]  # None when untimed
    end_sec: Optional[float]  # None when untimed
    text: str


_WORD_RE = re.compile(r"[A-Za-z']+")


def _safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _parse_timestamp_to_seconds(ts: str) -> float:
    """
    Supports:
      - HH:MM:SS,mmm   (SRT)
      - HH:MM:SS.mmm   (VTT)
      - MM:SS,mmm / MM:SS.mmm
    """
    s = ts.strip()
    s = s.replace(",", ".")
    parts = s.split(":")
    if len(parts) == 3:
        hh, mm, ss = parts
    elif len(parts) == 2:
        hh = "0"
        mm, ss = parts
    else:
        raise ValueError(f"Unrecognized timestamp: {ts!r}")
    return int(hh) * 3600 + int(mm) * 60 + float(ss)


def _strip_vtt_markup(line: str) -> str:
    # Remove simple tags like <c>, <i>, <b>, timestamps, etc.
    # Keep readable text only.
    line = re.sub(r"<[^>]+>", "", line)
    line = re.sub(r"\s+", " ", line).strip()
    return line


def load_srt_or_svt(path: Path) -> List[Segment]:
    text = _safe_read_text(path)
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n").strip())
    out: List[Segment] = []

    for block in blocks:
        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue

        # Many SRTs start with an index line; tolerate missing index.
        time_line_idx = 0
        if re.fullmatch(r"\d+", lines[0]):
            time_line_idx = 1
        if time_line_idx >= len(lines):
            continue

        time_line = lines[time_line_idx]
        if "-->" not in time_line:
            # Not a valid cue; skip.
            continue

        lhs, rhs = [p.strip() for p in time_line.split("-->", 1)]
        # Drop cue settings after end timestamp, e.g. "00:00:01.000 align:start"
        rhs = rhs.split()[0].strip()
        start = _parse_timestamp_to_seconds(lhs)
        end = _parse_timestamp_to_seconds(rhs)

        cue_text_lines = lines[time_line_idx + 1 :]
        cue_text = " ".join(cue_text_lines).strip()
        cue_text = re.sub(r"\s+", " ", cue_text).strip()
        if cue_text:
            out.append(Segment(start_sec=start, end_sec=end, text=cue_text))
    return out


def load_vtt(path: Path) -> List[Segment]:
    raw = _safe_read_text(path)
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    # Drop WEBVTT header and NOTE blocks.
    lines = raw.split("\n")
    cleaned: List[str] = []
    in_note = False
    for ln in lines:
        if ln.strip().startswith("NOTE"):
            in_note = True
            continue
        if in_note:
            if not ln.strip():
                in_note = False
            continue
        cleaned.append(ln)
    text = "\n".join(cleaned).strip()

    blocks = re.split(r"\n\s*\n", text)
    out: List[Segment] = []
    for block in blocks:
        bl = [ln.strip() for ln in block.split("\n") if ln.strip()]
        if not bl:
            continue
        # Skip header block if present
        if bl[0].upper().startswith("WEBVTT"):
            continue

        time_idx = 0
        if "-->" not in bl[0] and len(bl) > 1 and "-->" in bl[1]:
            # Optional cue identifier line.
            time_idx = 1
        if "-->" not in bl[time_idx]:
            continue

        time_line = bl[time_idx]
        lhs, rhs = [p.strip() for p in time_line.split("-->", 1)]
        rhs = rhs.split()[0].strip()
        start = _parse_timestamp_to_seconds(lhs)
        end = _parse_timestamp_to_seconds(rhs)

        cue_lines = [_strip_vtt_markup(ln) for ln in bl[time_idx + 1 :]]
        cue_text = " ".join([c for c in cue_lines if c]).strip()
        cue_text = re.sub(r"\s+", " ", cue_text).strip()
        if cue_text:
            out.append(Segment(start_sec=start, end_sec=end, text=cue_text))
    return out


def load_md_or_txt(path: Path) -> List[Segment]:
    text = _safe_read_text(path)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # If markdown, strip code fences and inline formatting (best-effort).
    if path.suffix.lower() in {".md", ".markdown"}:
        # Remove fenced code blocks
        text = re.sub(r"```[\s\S]*?```", " ", text)
        # Remove inline code
        text = re.sub(r"`[^`]*`", " ", text)
        # Remove links [text](url) -> text
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        # Remove images ![alt](url) -> alt
        text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
        # Remove headings markers
        text = re.sub(r"^\s{0,3}#{1,6}\s+", "", text, flags=re.MULTILINE)
        # Remove emphasis markers
        text = text.replace("*", " ").replace("_", " ")

    # Split into paragraph-ish segments (untimed).
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out: List[Segment] = []
    for p in paras:
        p = re.sub(r"\s+", " ", p).strip()
        if p:
            out.append(Segment(start_sec=None, end_sec=None, text=p))
    if not out and text.strip():
        out = [Segment(start_sec=None, end_sec=None, text=re.sub(r"\s+", " ", text).strip())]
    return out


def load_pdf(path: Path) -> List[Segment]:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency for PDF. Install: pip install pypdf"
        ) from e

    reader = PdfReader(str(path))
    pages: List[str] = []
    for pg in reader.pages:
        t = pg.extract_text() or ""
        t = re.sub(r"\s+", " ", t).strip()
        if t:
            pages.append(t)
    text = "\n\n".join(pages).strip()
    if not text:
        return []
    return [Segment(start_sec=None, end_sec=None, text=p) for p in pages] or [
        Segment(start_sec=None, end_sec=None, text=text)
    ]


def load_docx(path: Path) -> List[Segment]:
    try:
        import docx  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency for DOCX. Install: pip install python-docx"
        ) from e

    doc = docx.Document(str(path))
    paras = []
    for p in doc.paragraphs:
        t = re.sub(r"\s+", " ", (p.text or "")).strip()
        if t:
            paras.append(t)
    if not paras:
        return []
    return [Segment(start_sec=None, end_sec=None, text=t) for t in paras]


def load_rtf(path: Path) -> List[Segment]:
    try:
        from striprtf.striprtf import rtf_to_text  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency for RTF. Install: pip install striprtf"
        ) from e

    raw = _safe_read_text(path)
    text = rtf_to_text(raw)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return [Segment(start_sec=None, end_sec=None, text=p) for p in paras]


def load_whisper_json(path: Path) -> List[Segment]:
    """
    Supports common Whisper-style JSON:
      - {"text": "...", "segments":[{"start":0.0,"end":1.2,"text":"..."}]}
    """
    try:
        obj = json.loads(_safe_read_text(path))
    except Exception as e:
        raise RuntimeError(f"Invalid JSON: {e}") from e

    if isinstance(obj, dict) and isinstance(obj.get("segments"), list):
        out: List[Segment] = []
        for seg in obj["segments"]:
            if not isinstance(seg, dict):
                continue
            t = str(seg.get("text") or "").strip()
            if not t:
                continue
            start = seg.get("start")
            end = seg.get("end")
            start_f = float(start) if isinstance(start, (int, float)) else None
            end_f = float(end) if isinstance(end, (int, float)) else None
            out.append(Segment(start_sec=start_f, end_sec=end_f, text=re.sub(r"\s+", " ", t)))
        if out:
            return out

    if isinstance(obj, dict) and isinstance(obj.get("text"), str) and obj["text"].strip():
        return [Segment(start_sec=None, end_sec=None, text=re.sub(r"\s+", " ", obj["text"]).strip())]

    # Unknown JSON shape; fallback to plain-text-ish.
    return [Segment(start_sec=None, end_sec=None, text=_safe_read_text(path).strip())]


def load_csv_or_tsv(path: Path) -> List[Segment]:
    """
    Attempts to load rows containing text and optional start/end.

    Expected headers (case-insensitive):
      - text (required)
      - start/start_sec (optional)
      - end/end_sec (optional)
    """
    raw = _safe_read_text(path)
    dialect = csv.Sniffer().sniff(raw[:4096], delimiters=",\t;")
    reader = csv.DictReader(raw.splitlines(), dialect=dialect)
    if not reader.fieldnames:
        return load_md_or_txt(path)
    fields = {f.lower().strip(): f for f in reader.fieldnames if f}
    text_key = fields.get("text")
    if not text_key:
        return load_md_or_txt(path)

    start_key = fields.get("start") or fields.get("start_sec")
    end_key = fields.get("end") or fields.get("end_sec")

    out: List[Segment] = []
    for row in reader:
        t = (row.get(text_key) or "").strip()
        if not t:
            continue
        start_sec = None
        end_sec = None
        if start_key and row.get(start_key):
            try:
                start_sec = float(row[start_key])
            except Exception:
                start_sec = None
        if end_key and row.get(end_key):
            try:
                end_sec = float(row[end_key])
            except Exception:
                end_sec = None
        out.append(Segment(start_sec=start_sec, end_sec=end_sec, text=re.sub(r"\s+", " ", t)))
    return out or load_md_or_txt(path)


def load_any(path: Path) -> List[Segment]:
    ext = path.suffix.lower()
    if ext in {".srt", ".svt"}:
        return load_srt_or_svt(path)
    if ext == ".vtt":
        return load_vtt(path)
    if ext in {".txt", ".md", ".markdown"}:
        return load_md_or_txt(path)
    if ext == ".json":
        return load_whisper_json(path)
    if ext in {".csv", ".tsv"}:
        return load_csv_or_tsv(path)
    if ext == ".pdf":
        return load_pdf(path)
    if ext == ".docx":
        return load_docx(path)
    if ext == ".rtf":
        return load_rtf(path)

    # Fallback: treat as plain text.
    return load_md_or_txt(path)


def _tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


def _emotion_lexicons() -> dict[str, set[str]]:
    """
    Small, built-in lexicons for offline emotion scoring.
    Not exhaustive; intended for heatmap-like trend visualization.
    """
    return {
        "joy": {
            "happy",
            "joy",
            "glad",
            "excited",
            "thrilled",
            "delighted",
            "love",
            "loved",
            "awesome",
            "great",
            "fantastic",
            "wonderful",
            "smile",
            "laugh",
            "relieved",
            "proud",
        },
        "sadness": {
            "sad",
            "unhappy",
            "depressed",
            "down",
            "cry",
            "cried",
            "tears",
            "grief",
            "lonely",
            "heartbroken",
            "miserable",
            "regret",
            "sorry",
            "loss",
        },
        "anger": {
            "angry",
            "mad",
            "furious",
            "rage",
            "annoyed",
            "irritated",
            "frustrated",
            "hate",
            "hated",
            "upset",
            "pissed",
            "outraged",
        },
        "fear": {
            "afraid",
            "scared",
            "fear",
            "terrified",
            "anxious",
            "nervous",
            "worry",
            "worried",
            "panic",
            "panicked",
            "unsafe",
            "danger",
            "dread",
        },
        "disgust": {
            "disgust",
            "disgusted",
            "gross",
            "nasty",
            "sick",
            "repulsed",
            "awful",
            "horrible",
        },
        "surprise": {
            "surprised",
            "shocked",
            "wow",
            "unexpected",
            "suddenly",
            "amazed",
            "astonished",
        },
        "trust": {
            "trust",
            "trusted",
            "safe",
            "secure",
            "confident",
            "reliable",
            "faith",
            "honest",
        },
        "anticipation": {
            "hope",
            "hopeful",
            "expect",
            "expected",
            "anticipate",
            "anticipation",
            "looking",
            "forward",
            "soon",
            "plan",
        },
    }


def _merge_lexicons(base: dict[str, set[str]], extra: dict[str, Iterable[str]]) -> dict[str, set[str]]:
    merged = {k: set(v) for k, v in base.items()}
    for emo, words in extra.items():
        merged.setdefault(emo, set()).update({str(w).lower() for w in words if str(w).strip()})
    return merged


def score_emotions(text: str, emotions: Sequence[str], lexicon: dict[str, set[str]]) -> dict[str, float]:
    toks = _tokenize(text)
    if not toks:
        return {e: 0.0 for e in emotions}

    counts = {e: 0 for e in emotions}
    for t in toks:
        for e in emotions:
            if t in lexicon.get(e, set()):
                counts[e] += 1

    denom = max(1, len(toks))
    # Normalize as "hits per 100 tokens" to be human-readable.
    return {e: (counts[e] / denom) * 100.0 for e in emotions}


def _bin_timed_segments(segments: Sequence[Segment], bin_seconds: float) -> List[Segment]:
    timed = [s for s in segments if s.start_sec is not None and s.end_sec is not None]
    if not timed:
        return list(segments)
    if bin_seconds <= 0:
        return list(segments)

    start0 = min(s.start_sec for s in timed if s.start_sec is not None)
    end_max = max(s.end_sec for s in timed if s.end_sec is not None)
    n_bins = int(math.ceil((end_max - start0) / bin_seconds))
    if n_bins <= 0:
        return list(segments)

    bins: List[List[str]] = [[] for _ in range(n_bins)]
    eps = 1e-9  # avoid double-counting cues that end exactly on bin edge
    for s in timed:
        # Place cue text into bins it overlaps (simple approach).
        s0 = s.start_sec or 0.0
        s1 = s.end_sec or s0
        b0 = int((s0 - start0) // bin_seconds)
        s1_adj = max(s0, s1) - eps
        b1 = int((s1_adj - start0) // bin_seconds)
        b0 = max(0, min(n_bins - 1, b0))
        b1 = max(0, min(n_bins - 1, b1))
        for b in range(b0, b1 + 1):
            bins[b].append(s.text)

    out: List[Segment] = []
    for i, texts in enumerate(bins):
        b_start = start0 + i * bin_seconds
        b_end = b_start + bin_seconds
        joined = re.sub(r"\s+", " ", " ".join(texts)).strip()
        out.append(Segment(start_sec=b_start, end_sec=b_end, text=joined))
    return out


def _bin_untimed_segments(segments: Sequence[Segment], bin_words: int) -> List[Segment]:
    if bin_words <= 0:
        return list(segments)
    text = " ".join(s.text for s in segments)
    toks = _tokenize(text)
    if not toks:
        return list(segments)

    out: List[Segment] = []
    for i in range(0, len(toks), bin_words):
        chunk = " ".join(toks[i : i + bin_words])
        out.append(Segment(start_sec=None, end_sec=None, text=chunk))
    return out


def write_csv(path: Path, bins: Sequence[Segment], rows: Sequence[dict[str, float]], emotions: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["bin", "start_sec", "end_sec", "text"] + list(emotions)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, (b, r) in enumerate(zip(bins, rows)):
            row = {
                "bin": i,
                "start_sec": "" if b.start_sec is None else f"{b.start_sec:.3f}",
                "end_sec": "" if b.end_sec is None else f"{b.end_sec:.3f}",
                "text": b.text,
            }
            row.update({e: f"{r[e]:.6f}" for e in emotions})
            w.writerow(row)


def write_json(path: Path, bins: Sequence[Segment], rows: Sequence[dict[str, float]], emotions: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for i, (b, r) in enumerate(zip(bins, rows)):
        payload.append(
            {
                "bin": i,
                "start_sec": b.start_sec,
                "end_sec": b.end_sec,
                "text": b.text,
                "emotions": {e: r[e] for e in emotions},
            }
        )
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_heatmap_png(path: Path, rows: Sequence[dict[str, float]], emotions: Sequence[str], title: str) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency for plotting. Install: pip install matplotlib numpy") from e

    if not rows:
        raise RuntimeError("No bins/rows to plot.")

    data = np.array([[r[e] for r in rows] for e in emotions], dtype=float)

    # Robust color scaling: clip at 95th percentile.
    vmax = float(np.percentile(data, 95)) if np.any(data) else 1.0
    vmax = max(vmax, 1e-6)

    fig_w = max(10.0, min(36.0, 0.18 * data.shape[1] + 4.0))
    fig_h = max(4.0, 0.45 * data.shape[0] + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)

    im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap="magma", vmin=0.0, vmax=vmax)
    ax.set_yticks(range(len(emotions)))
    ax.set_yticklabels(emotions)
    ax.set_xticks([])
    ax.set_xlabel("time/sequence bins")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("emotion intensity (hits per 100 tokens)")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Create an emotional heatmap from a transcription file (SRT/VTT/TXT/PDF/RTF/MD/DOCX/...)."
    )
    ap.add_argument("input", help="Path to transcript file")
    ap.add_argument("--out-dir", default="out", help="Output directory (default: out/)")
    ap.add_argument(
        "--emotions",
        default="joy,sadness,anger,fear,disgust,surprise,trust,anticipation",
        help="Comma-separated list of emotions to score (default: 8 basic emotions)",
    )
    ap.add_argument(
        "--lexicon",
        default=None,
        help=(
            "Optional path to a JSON lexicon mapping emotion->list of words. "
            "Merged into the built-in lexicon."
        ),
    )
    ap.add_argument(
        "--bin-seconds",
        type=float,
        default=30.0,
        help="Bin size in seconds for timed transcripts (default: 30). Ignored for untimed inputs.",
    )
    ap.add_argument(
        "--bin-words",
        type=int,
        default=120,
        help="Bin size in words for untimed inputs (default: 120).",
    )
    ap.add_argument(
        "--min-text-len",
        type=int,
        default=1,
        help="Drop bins with fewer than this many characters after binning (default: 1).",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    in_path = Path(args.input).expanduser().resolve()
    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir).expanduser().resolve()
    emotions = [e.strip() for e in args.emotions.split(",") if e.strip()]
    if not emotions:
        print("ERROR: --emotions resulted in empty list", file=sys.stderr)
        return 2

    try:
        segments = load_any(in_path)
    except Exception as e:
        print(f"ERROR: failed to load input: {e}", file=sys.stderr)
        return 2

    if not segments:
        print("ERROR: no text segments found in input", file=sys.stderr)
        return 2

    # Choose binning strategy.
    has_timing = any(s.start_sec is not None and s.end_sec is not None for s in segments)
    if has_timing:
        bins = _bin_timed_segments(segments, bin_seconds=float(args.bin_seconds))
    else:
        bins = _bin_untimed_segments(segments, bin_words=int(args.bin_words))

    bins = [b for b in bins if len((b.text or "").strip()) >= int(args.min_text_len)]
    if not bins:
        print("ERROR: all bins filtered out (try lowering --min-text-len)", file=sys.stderr)
        return 2

    lexicon = _emotion_lexicons()
    if args.lexicon:
        lex_path = Path(args.lexicon).expanduser().resolve()
        try:
            extra = json.loads(_safe_read_text(lex_path))
            if not isinstance(extra, dict):
                raise ValueError("lexicon JSON must be an object mapping emotion->words")
            lexicon = _merge_lexicons(
                lexicon,
                {str(k): (v if isinstance(v, list) else [v]) for k, v in extra.items()},
            )
        except Exception as e:
            print(f"ERROR: failed to load --lexicon JSON: {e}", file=sys.stderr)
            return 2

    rows = [score_emotions(b.text, emotions=emotions, lexicon=lexicon) for b in bins]

    stem = in_path.stem
    csv_path = out_dir / f"{stem}.emotion_bins.csv"
    png_path = out_dir / f"{stem}.emotion_heatmap.png"
    json_path = out_dir / f"{stem}.emotion_bins.json"

    write_csv(csv_path, bins=bins, rows=rows, emotions=emotions)
    write_json(json_path, bins=bins, rows=rows, emotions=emotions)
    title = f"Emotional heatmap: {in_path.name} ({'timed' if has_timing else 'untimed'})"
    write_heatmap_png(png_path, rows=rows, emotions=emotions, title=title)

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

