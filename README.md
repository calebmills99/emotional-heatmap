# Emotional Heatmap Generator for Transcriptions

A powerful Python script for post-transcription analysis that generates emotional heatmaps from transcription files. Analyze the emotional journey of interviews, podcasts, meetings, or any spoken content.

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)

## Features

- **Multi-format Support**: Process transcriptions in various formats:
  - **Text**: `.txt`, `.md` (Markdown)
  - **Documents**: `.docx`, `.pdf`, `.rtf`, `.odt`, `.epub`
  - **Subtitles**: `.srt`, `.vtt`, `.sbv`
  - **Data**: `.json`, `.csv`
  - **Web**: `.html`, `.htm`

- **Multiple Visualization Styles**:
  - **Heatmap**: Traditional 2D heatmap showing emotion intensity over segments
  - **Timeline**: Stacked area chart with dominant emotion markers
  - **Radar**: Spider/radar chart showing overall emotion distribution
  - **Stream**: Streamgraph showing emotion flow over time

- **Flexible Emotion Analysis**:
  - Built-in basic lexicon (no external dependencies)
  - NRC Emotion Lexicon integration (recommended)
  - Transformer-based models for highest accuracy (optional)

- **Detailed Reporting**: Generate JSON reports with segment-by-segment analysis

## Installation

### Quick Install

```bash
# Clone or download the script
git clone <repository-url>
cd emotional-heatmap

# Install dependencies
pip install -r requirements.txt
```

### Minimal Install (Basic Features Only)

```bash
pip install numpy matplotlib seaborn scipy
```

### Full Install (All Features)

```bash
pip install -r requirements.txt
```

### With Transformer Support (Best Accuracy)

```bash
pip install -r requirements.txt
pip install transformers torch
```

## Usage

### Basic Usage

```bash
# Generate heatmap from a text file
python emotional_heatmap.py transcript.txt

# Generate heatmap from various formats
python emotional_heatmap.py interview.docx
python emotional_heatmap.py podcast.srt
python emotional_heatmap.py meeting.pdf
```

### Specify Output File

```bash
python emotional_heatmap.py transcript.txt -o my_analysis.png
```

### Different Visualization Styles

```bash
# Traditional heatmap (default)
python emotional_heatmap.py transcript.txt --style heatmap

# Timeline with stacked emotions
python emotional_heatmap.py transcript.txt --style timeline

# Radar/spider chart of overall emotions
python emotional_heatmap.py transcript.txt --style radar

# Streamgraph visualization
python emotional_heatmap.py transcript.txt --style stream
```

### Generate JSON Report

```bash
python emotional_heatmap.py transcript.txt --report
```

This creates both the visualization and a detailed JSON report with:
- Overall emotion distribution percentages
- Segment-by-segment emotion analysis
- Dominant emotions and intensity scores

### Use Transformer Model (Higher Accuracy)

```bash
python emotional_heatmap.py transcript.txt --transformer
```

### Custom Title

```bash
python emotional_heatmap.py interview.txt --title "Customer Interview - Jan 2024"
```

### List Supported Formats

```bash
python emotional_heatmap.py --list-formats
```

## Command Line Options

| Option | Description |
|--------|-------------|
| `input_file` | Input transcription file (required) |
| `-o, --output` | Output image file (default: `<input>_heatmap.png`) |
| `--style` | Visualization style: `heatmap`, `timeline`, `radar`, `stream` |
| `--transformer` | Use transformer model for better accuracy |
| `--model` | Specific transformer model to use |
| `--report` | Also generate a JSON report |
| `--title` | Custom title for the visualization |
| `--segment-size` | Target segment size in characters (default: 200) |
| `--list-formats` | List supported file formats and exit |

## Input File Formats

### Plain Text (.txt)
Simple text files with transcription content.

### Markdown (.md)
Markdown files - formatting is stripped for analysis.

### Microsoft Word (.docx)
Word documents with text content.

### PDF (.pdf)
PDF documents - text is extracted from all pages.

### Rich Text (.rtf)
RTF documents with formatting stripped.

### Subtitles (.srt, .vtt, .sbv)
Subtitle files preserve timestamps for accurate segment mapping:
```
1
00:00:01,000 --> 00:00:04,000
Hello, welcome to the interview.

2
00:00:04,500 --> 00:00:08,000
Thank you for having me, I'm excited to be here!
```

### JSON Transcripts (.json)
Supports various JSON formats including:
```json
[
  {"timestamp": "00:00:01", "speaker": "Host", "text": "Welcome everyone!"},
  {"timestamp": "00:00:05", "speaker": "Guest", "text": "Thanks for having me."}
]
```

### CSV Files (.csv)
CSV with columns like `text`, `timestamp`, `speaker`:
```csv
timestamp,speaker,text
00:00:01,Host,Welcome to the show
00:00:05,Guest,Thank you for having me
```

## Emotion Categories

The analyzer detects the following emotions:

| Emotion | Color | Description |
|---------|-------|-------------|
| Joy | Gold | Happiness, delight, pleasure |
| Sadness | Royal Blue | Sorrow, grief, melancholy |
| Anger | Crimson | Frustration, hostility, rage |
| Fear | Purple | Anxiety, worry, dread |
| Surprise | Hot Pink | Astonishment, amazement |
| Disgust | Saddle Brown | Revulsion, distaste |
| Trust | Lime Green | Confidence, faith, reliability |
| Anticipation | Dark Orange | Expectation, hope, eagerness |

## Output Examples

### Heatmap Style
Shows emotion intensity (color) for each emotion (y-axis) across text segments (x-axis).

### Timeline Style
Two-panel visualization:
- Top: Stacked area chart showing all emotions over time
- Bottom: Scatter plot of dominant emotions

### Radar Style
Spider chart showing the overall distribution of emotions across the entire transcription.

### Stream Style
Streamgraph showing the flow and interplay of emotions throughout the transcription.

## JSON Report Format

```json
{
  "summary": {
    "total_segments": 45,
    "emotions_detected": ["joy", "sadness", "anger", "fear", "trust"]
  },
  "emotion_distribution": {
    "joy": 35.5,
    "trust": 28.3,
    "anticipation": 15.2,
    "sadness": 12.0,
    "fear": 9.0
  },
  "segments": [
    {
      "id": 0,
      "text": "Hello and welcome to today's interview...",
      "timestamp": "00:00:01",
      "speaker": "Host",
      "dominant_emotion": "joy",
      "intensity": 0.65,
      "emotions": {
        "joy": 0.65,
        "trust": 0.25,
        "anticipation": 0.10
      }
    }
  ]
}
```

## Tips for Best Results

1. **Clean transcriptions**: Better input text = better emotion detection
2. **Longer segments**: Very short segments may not have enough context
3. **Use transformer mode**: For critical analysis, use `--transformer` for higher accuracy
4. **Check the report**: Use `--report` to understand segment-by-segment emotions
5. **Subtitle formats**: These preserve timing information, great for video analysis

## Troubleshooting

### Missing Dependencies
```bash
# Install missing format support
pip install pdfplumber      # For PDF
pip install python-docx     # For DOCX
pip install striprtf        # For RTF
pip install nrclex          # For better emotion detection
```

### Memory Issues with Large Files
- Use `--segment-size` to adjust segment granularity
- For very large files, consider splitting them

### Transformer Model Issues
```bash
# Ensure you have enough memory and install:
pip install transformers torch
```

## License

MIT License - Feel free to use, modify, and distribute.

## Contributing

Contributions welcome! Please feel free to submit issues and pull requests.
