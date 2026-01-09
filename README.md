# whisper_auto

Utilities for whisper/transcription automation.

## Emotional heatmap (post-transcription)

The script `scripts/emotional_heatmap.py` takes a transcription file and produces:

- a **CSV** of binned segments with emotion scores
- a **PNG heatmap** (emotions × time/sequence bins)
- a **JSON** export of the same bins/scores

### Supported input formats

- **Subtitles**: `.srt`, `.vtt` (and `.svt` treated like `.srt`)
- **Text/docs**: `.txt`, `.md`, `.pdf`, `.rtf`, `.docx`
- **Structured**: `.json` (Whisper-style segments), `.csv` / `.tsv` (text + optional start/end)
- **Fallback**: any other file is treated as plain text (UTF-8 with replacement)

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
python3 scripts/emotional_heatmap.py "path/to/transcript.srt" --out-dir out --bin-seconds 30
```

Untimed documents (TXT/MD/PDF/DOCX/RTF) are binned by words:

```bash
python3 scripts/emotional_heatmap.py "notes.docx" --out-dir out --bin-words 120
```

To customize the emotion wordlists, provide a JSON lexicon:

```bash
python3 scripts/emotional_heatmap.py "transcript.vtt" --lexicon lexicon.json --out-dir out
```

Outputs are written to `--out-dir` as:

- `<stem>.emotion_bins.csv`
- `<stem>.emotion_bins.json`
- `<stem>.emotion_heatmap.png`
