#!/usr/bin/env python3
"""
Emotional Heatmap Generator for Transcription Files

This script analyzes transcription files and generates an emotional heatmap
visualization showing the distribution and intensity of emotions throughout the text.

Supported formats: TXT, PDF, RTF, MD, DOCX, SRT, VTT, SBV, JSON, CSV, HTML, ODT, EPUB
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import seaborn as sns

# Text extraction libraries
try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from striprtf.striprtf import rtf_to_text
except ImportError:
    rtf_to_text = None

try:
    import markdown
    from bs4 import BeautifulSoup
except ImportError:
    markdown = None
    BeautifulSoup = None

try:
    import webvtt
except ImportError:
    webvtt = None

try:
    import pysrt
except ImportError:
    pysrt = None

try:
    from odf import text as odf_text
    from odf.opendocument import load as odf_load
except ImportError:
    odf_text = None
    odf_load = None

try:
    import ebooklib
    from ebooklib import epub
except ImportError:
    ebooklib = None
    epub = None

try:
    from nrclex import NRCLex
except ImportError:
    NRCLex = None

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# Emotion categories and their associated colors
EMOTION_COLORS = {
    'joy': '#FFD700',           # Gold
    'happiness': '#FFD700',     # Gold
    'trust': '#32CD32',         # Lime Green
    'fear': '#800080',          # Purple
    'surprise': '#FF69B4',      # Hot Pink
    'sadness': '#4169E1',       # Royal Blue
    'disgust': '#8B4513',       # Saddle Brown
    'anger': '#DC143C',         # Crimson
    'anticipation': '#FF8C00',  # Dark Orange
    'positive': '#00CED1',      # Dark Turquoise
    'negative': '#B22222',      # Fire Brick
    'neutral': '#808080',       # Gray
    'love': '#FF1493',          # Deep Pink
    'optimism': '#98FB98',      # Pale Green
    'pessimism': '#696969',     # Dim Gray
}

# Fallback emotion lexicon if NRCLex is not available
BASIC_EMOTION_LEXICON = {
    'joy': ['happy', 'joy', 'delighted', 'pleased', 'glad', 'cheerful', 'elated', 
            'thrilled', 'ecstatic', 'blissful', 'wonderful', 'fantastic', 'great',
            'amazing', 'excellent', 'perfect', 'beautiful', 'love', 'loving', 'loved'],
    'sadness': ['sad', 'unhappy', 'depressed', 'miserable', 'sorrowful', 'gloomy',
                'melancholy', 'heartbroken', 'devastated', 'disappointed', 'upset',
                'crying', 'tears', 'grief', 'mourning', 'tragic', 'unfortunate'],
    'anger': ['angry', 'mad', 'furious', 'enraged', 'irritated', 'annoyed', 'frustrated',
              'outraged', 'hostile', 'hate', 'hatred', 'rage', 'livid', 'bitter'],
    'fear': ['afraid', 'scared', 'frightened', 'terrified', 'anxious', 'worried',
             'nervous', 'panic', 'dread', 'horror', 'alarmed', 'threatened'],
    'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'stunned', 'startled',
                 'unexpected', 'incredible', 'unbelievable', 'wow', 'whoa'],
    'disgust': ['disgusted', 'revolted', 'repulsed', 'sickened', 'nauseated', 'gross',
                'awful', 'terrible', 'horrible', 'dreadful', 'appalling'],
    'trust': ['trust', 'believe', 'faith', 'confident', 'reliable', 'honest', 'loyal',
              'sincere', 'genuine', 'dependable', 'faithful'],
    'anticipation': ['anticipate', 'expect', 'await', 'hope', 'eager', 'excited',
                     'looking forward', 'curious', 'interested', 'wonder'],
}


@dataclass
class TextSegment:
    """Represents a segment of text with its position and content."""
    text: str
    start_pos: int
    end_pos: int
    timestamp: Optional[str] = None
    speaker: Optional[str] = None


@dataclass
class EmotionScore:
    """Represents emotion scores for a text segment."""
    segment_id: int
    emotions: Dict[str, float] = field(default_factory=dict)
    dominant_emotion: str = 'neutral'
    intensity: float = 0.0


class TextExtractor:
    """Handles extraction of text from various file formats."""
    
    SUPPORTED_FORMATS = {
        '.txt': 'text',
        '.pdf': 'pdf',
        '.rtf': 'rtf',
        '.md': 'markdown',
        '.docx': 'docx',
        '.doc': 'docx',
        '.srt': 'srt',
        '.vtt': 'vtt',
        '.sbv': 'sbv',
        '.json': 'json',
        '.csv': 'csv',
        '.html': 'html',
        '.htm': 'html',
        '.odt': 'odt',
        '.epub': 'epub',
    }
    
    def __init__(self, file_path: str, segment_size: int = 200):
        self.file_path = Path(file_path)
        self.extension = self.file_path.suffix.lower()
        self.segment_size = segment_size
        
    def extract(self) -> List[TextSegment]:
        """Extract text segments from the file."""
        if self.extension not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {self.extension}")
        
        format_type = self.SUPPORTED_FORMATS[self.extension]
        extractor_method = getattr(self, f'_extract_{format_type}', None)
        
        if extractor_method is None:
            raise NotImplementedError(f"Extractor for {format_type} not implemented")
        
        return extractor_method()
    
    def _extract_text(self) -> List[TextSegment]:
        """Extract from plain text files."""
        with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return self._split_into_segments(content)
    
    def _extract_pdf(self) -> List[TextSegment]:
        """Extract from PDF files."""
        if pdfplumber is None:
            raise ImportError("pdfplumber is required for PDF support. Install with: pip install pdfplumber")
        
        segments = []
        pos = 0
        with pdfplumber.open(self.file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                if text.strip():
                    segments.extend(self._split_into_segments(text, start_pos=pos))
                    pos += len(text)
        return segments
    
    def _extract_rtf(self) -> List[TextSegment]:
        """Extract from RTF files."""
        if rtf_to_text is None:
            raise ImportError("striprtf is required for RTF support. Install with: pip install striprtf")
        
        with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            rtf_content = f.read()
        content = rtf_to_text(rtf_content)
        return self._split_into_segments(content)
    
    def _extract_markdown(self) -> List[TextSegment]:
        """Extract from Markdown files."""
        with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        if markdown is not None and BeautifulSoup is not None:
            html = markdown.markdown(content)
            soup = BeautifulSoup(html, 'html.parser')
            content = soup.get_text()
        
        return self._split_into_segments(content)
    
    def _extract_docx(self) -> List[TextSegment]:
        """Extract from DOCX files."""
        if DocxDocument is None:
            raise ImportError("python-docx is required for DOCX support. Install with: pip install python-docx")
        
        doc = DocxDocument(self.file_path)
        content = '\n'.join([para.text for para in doc.paragraphs])
        return self._split_into_segments(content)
    
    def _extract_srt(self) -> List[TextSegment]:
        """Extract from SRT subtitle files."""
        if pysrt is None:
            # Fallback to manual parsing
            return self._parse_srt_manual()
        
        subs = pysrt.open(self.file_path)
        segments = []
        pos = 0
        for sub in subs:
            text = sub.text.replace('\n', ' ')
            timestamp = f"{sub.start} --> {sub.end}"
            segments.append(TextSegment(
                text=text,
                start_pos=pos,
                end_pos=pos + len(text),
                timestamp=timestamp
            ))
            pos += len(text) + 1
        return segments
    
    def _parse_srt_manual(self) -> List[TextSegment]:
        """Manual SRT parsing fallback."""
        with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        segments = []
        pos = 0
        # SRT format: index, timestamp, text, blank line
        blocks = re.split(r'\n\n+', content.strip())
        
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                timestamp = lines[1]
                text = ' '.join(lines[2:])
                segments.append(TextSegment(
                    text=text,
                    start_pos=pos,
                    end_pos=pos + len(text),
                    timestamp=timestamp
                ))
                pos += len(text) + 1
        return segments
    
    def _extract_vtt(self) -> List[TextSegment]:
        """Extract from WebVTT subtitle files."""
        if webvtt is not None:
            try:
                segments = []
                pos = 0
                for caption in webvtt.read(str(self.file_path)):
                    text = caption.text.replace('\n', ' ')
                    timestamp = f"{caption.start} --> {caption.end}"
                    segments.append(TextSegment(
                        text=text,
                        start_pos=pos,
                        end_pos=pos + len(text),
                        timestamp=timestamp
                    ))
                    pos += len(text) + 1
                return segments
            except Exception:
                pass
        
        # Fallback to manual parsing
        return self._parse_vtt_manual()
    
    def _parse_vtt_manual(self) -> List[TextSegment]:
        """Manual VTT parsing fallback."""
        with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        segments = []
        pos = 0
        lines = content.split('\n')
        i = 0
        
        # Skip WEBVTT header
        while i < len(lines) and not re.match(r'\d{2}:\d{2}', lines[i]):
            i += 1
        
        while i < len(lines):
            if re.match(r'\d{2}:\d{2}', lines[i]):
                timestamp = lines[i]
                i += 1
                text_lines = []
                while i < len(lines) and lines[i].strip():
                    text_lines.append(lines[i])
                    i += 1
                text = ' '.join(text_lines)
                if text:
                    segments.append(TextSegment(
                        text=text,
                        start_pos=pos,
                        end_pos=pos + len(text),
                        timestamp=timestamp
                    ))
                    pos += len(text) + 1
            i += 1
        return segments
    
    def _extract_sbv(self) -> List[TextSegment]:
        """Extract from SBV (YouTube) subtitle files."""
        with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        segments = []
        pos = 0
        blocks = re.split(r'\n\n+', content.strip())
        
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 2:
                timestamp = lines[0]
                text = ' '.join(lines[1:])
                segments.append(TextSegment(
                    text=text,
                    start_pos=pos,
                    end_pos=pos + len(text),
                    timestamp=timestamp
                ))
                pos += len(text) + 1
        return segments
    
    def _extract_json(self) -> List[TextSegment]:
        """Extract from JSON transcription files."""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        segments = []
        pos = 0
        
        # Handle various JSON transcript formats
        if isinstance(data, list):
            for item in data:
                text = item.get('text', item.get('content', item.get('transcript', str(item))))
                timestamp = item.get('timestamp', item.get('start', item.get('time', None)))
                speaker = item.get('speaker', item.get('name', None))
                segments.append(TextSegment(
                    text=str(text),
                    start_pos=pos,
                    end_pos=pos + len(str(text)),
                    timestamp=str(timestamp) if timestamp else None,
                    speaker=str(speaker) if speaker else None
                ))
                pos += len(str(text)) + 1
        elif isinstance(data, dict):
            # Try common transcript JSON structures
            transcript_keys = ['transcript', 'text', 'content', 'segments', 'results', 'utterances']
            for key in transcript_keys:
                if key in data:
                    if isinstance(data[key], list):
                        return self._process_json_list(data[key])
                    elif isinstance(data[key], str):
                        return self._split_into_segments(data[key])
            # Fallback: convert entire dict to string
            return self._split_into_segments(json.dumps(data, indent=2))
        
        return segments if segments else self._split_into_segments(str(data))
    
    def _process_json_list(self, items: List) -> List[TextSegment]:
        """Process a list of items from JSON."""
        segments = []
        pos = 0
        for item in items:
            if isinstance(item, dict):
                text = item.get('text', item.get('content', item.get('transcript', str(item))))
                timestamp = item.get('timestamp', item.get('start', item.get('time', None)))
                speaker = item.get('speaker', item.get('name', None))
            else:
                text = str(item)
                timestamp = None
                speaker = None
            
            segments.append(TextSegment(
                text=str(text),
                start_pos=pos,
                end_pos=pos + len(str(text)),
                timestamp=str(timestamp) if timestamp else None,
                speaker=str(speaker) if speaker else None
            ))
            pos += len(str(text)) + 1
        return segments
    
    def _extract_csv(self) -> List[TextSegment]:
        """Extract from CSV transcription files."""
        import csv
        
        segments = []
        pos = 0
        
        with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            # Try to detect if file has header
            sample = f.read(1024)
            f.seek(0)
            has_header = csv.Sniffer().has_header(sample) if sample else False
            
            reader = csv.DictReader(f) if has_header else csv.reader(f)
            
            for row in reader:
                if isinstance(row, dict):
                    # Look for text column
                    text_keys = ['text', 'content', 'transcript', 'message', 'body']
                    text = None
                    for key in text_keys:
                        if key in row:
                            text = row[key]
                            break
                    if text is None:
                        text = ' '.join(str(v) for v in row.values())
                    
                    timestamp = row.get('timestamp', row.get('time', row.get('start', None)))
                    speaker = row.get('speaker', row.get('name', None))
                else:
                    text = ' '.join(str(cell) for cell in row)
                    timestamp = None
                    speaker = None
                
                if text and text.strip():
                    segments.append(TextSegment(
                        text=text,
                        start_pos=pos,
                        end_pos=pos + len(text),
                        timestamp=str(timestamp) if timestamp else None,
                        speaker=str(speaker) if speaker else None
                    ))
                    pos += len(text) + 1
        
        return segments
    
    def _extract_html(self) -> List[TextSegment]:
        """Extract from HTML files."""
        with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        if BeautifulSoup is not None:
            soup = BeautifulSoup(content, 'html.parser')
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            content = soup.get_text()
        else:
            # Basic HTML tag removal
            content = re.sub(r'<[^>]+>', '', content)
        
        return self._split_into_segments(content)
    
    def _extract_odt(self) -> List[TextSegment]:
        """Extract from ODT (OpenDocument) files."""
        if odf_load is None or odf_text is None:
            raise ImportError("odfpy is required for ODT support. Install with: pip install odfpy")
        
        def get_text_content(element) -> str:
            """Recursively extract text content from ODF elements."""
            text_parts = []
            if hasattr(element, 'childNodes'):
                for child in element.childNodes:
                    if child.nodeType == child.TEXT_NODE:
                        text_parts.append(child.data)
                    else:
                        text_parts.append(get_text_content(child))
            return ''.join(text_parts)
        
        doc = odf_load(str(self.file_path))
        paragraphs = doc.getElementsByType(odf_text.P)
        content = '\n'.join([get_text_content(p) for p in paragraphs])
        return self._split_into_segments(content)
    
    def _extract_epub(self) -> List[TextSegment]:
        """Extract from EPUB files."""
        if epub is None:
            raise ImportError("ebooklib is required for EPUB support. Install with: pip install ebooklib")
        
        book = epub.read_epub(str(self.file_path))
        content_parts = []
        
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                if BeautifulSoup is not None:
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    content_parts.append(soup.get_text())
                else:
                    content_parts.append(re.sub(r'<[^>]+>', '', item.get_content().decode('utf-8', errors='ignore')))
        
        content = '\n'.join(content_parts)
        return self._split_into_segments(content)
    
    def _split_into_segments(self, content: str, start_pos: int = 0, 
                            segment_size: Optional[int] = None) -> List[TextSegment]:
        """Split content into analyzable segments."""
        # Use instance segment_size if not explicitly provided
        if segment_size is None:
            segment_size = self.segment_size
        
        # Clean the content
        content = re.sub(r'\s+', ' ', content).strip()
        
        if not content:
            return []
        
        segments = []
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        current_segment = []
        current_length = 0
        pos = start_pos
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if current_length + len(sentence) > segment_size and current_segment:
                segment_text = ' '.join(current_segment)
                segments.append(TextSegment(
                    text=segment_text,
                    start_pos=pos,
                    end_pos=pos + len(segment_text)
                ))
                pos += len(segment_text) + 1
                current_segment = [sentence]
                current_length = len(sentence)
            else:
                current_segment.append(sentence)
                current_length += len(sentence)
        
        # Add remaining content
        if current_segment:
            segment_text = ' '.join(current_segment)
            segments.append(TextSegment(
                text=segment_text,
                start_pos=pos,
                end_pos=pos + len(segment_text)
            ))
        
        return segments


class EmotionAnalyzer:
    """Analyzes emotions in text segments."""
    
    def __init__(self, use_transformer: bool = False, model_name: str = None):
        self.use_transformer = use_transformer and TRANSFORMERS_AVAILABLE
        self.classifier = None
        
        if self.use_transformer:
            model = model_name or "j-hartmann/emotion-english-distilroberta-base"
            try:
                self.classifier = pipeline("text-classification", 
                                         model=model,
                                         top_k=None,
                                         truncation=True)
            except Exception as e:
                print(f"Warning: Could not load transformer model: {e}")
                print("Falling back to lexicon-based analysis.")
                self.use_transformer = False
    
    def analyze(self, segments: List[TextSegment]) -> List[EmotionScore]:
        """Analyze emotions in all segments."""
        if self.use_transformer:
            return self._analyze_transformer(segments)
        elif NRCLex is not None:
            return self._analyze_nrclex(segments)
        else:
            return self._analyze_basic(segments)
    
    def _analyze_transformer(self, segments: List[TextSegment]) -> List[EmotionScore]:
        """Analyze using transformer model with batched processing for better performance."""
        scores = []
        
        # Prepare all texts for batch processing (truncate to 512 chars for model)
        texts = [segment.text[:512] for segment in segments]
        
        try:
            # Process all segments in a single batch call for efficiency
            results = self.classifier(texts)
            
            for i, result in enumerate(results):
                try:
                    emotions = {}
                    
                    if result and isinstance(result, list):
                        for item in result:
                            label = item['label'].lower()
                            score = item['score']
                            emotions[label] = score
                    
                    if emotions:
                        dominant = max(emotions, key=emotions.get)
                        intensity = emotions[dominant]
                    else:
                        dominant = 'neutral'
                        intensity = 0.0
                    
                    scores.append(EmotionScore(
                        segment_id=i,
                        emotions=emotions,
                        dominant_emotion=dominant,
                        intensity=intensity
                    ))
                except Exception:
                    scores.append(EmotionScore(
                        segment_id=i,
                        emotions={'neutral': 1.0},
                        dominant_emotion='neutral',
                        intensity=0.0
                    ))
        except Exception:
            # Fallback to individual processing if batch fails
            for i, segment in enumerate(segments):
                try:
                    result = self.classifier(segment.text[:512])
                    emotions = {}
                    
                    if result and isinstance(result, list):
                        for item in result[0] if isinstance(result[0], list) else result:
                            label = item['label'].lower()
                            score = item['score']
                            emotions[label] = score
                    
                    if emotions:
                        dominant = max(emotions, key=emotions.get)
                        intensity = emotions[dominant]
                    else:
                        dominant = 'neutral'
                        intensity = 0.0
                    
                    scores.append(EmotionScore(
                        segment_id=i,
                        emotions=emotions,
                        dominant_emotion=dominant,
                        intensity=intensity
                    ))
                except Exception:
                    scores.append(EmotionScore(
                        segment_id=i,
                        emotions={'neutral': 1.0},
                        dominant_emotion='neutral',
                        intensity=0.0
                    ))
        
        return scores
    
    def _analyze_nrclex(self, segments: List[TextSegment]) -> List[EmotionScore]:
        """Analyze using NRCLex lexicon."""
        scores = []
        
        for i, segment in enumerate(segments):
            emotion = NRCLex(segment.text)
            raw_scores = emotion.raw_emotion_scores
            
            # Normalize scores
            total = sum(raw_scores.values()) if raw_scores else 0
            emotions = {}
            
            if total > 0:
                for emo, score in raw_scores.items():
                    emotions[emo] = score / total
            else:
                emotions = {'neutral': 1.0}
            
            if emotions:
                dominant = max(emotions, key=emotions.get)
                intensity = emotions[dominant]
            else:
                dominant = 'neutral'
                intensity = 0.0
            
            scores.append(EmotionScore(
                segment_id=i,
                emotions=emotions,
                dominant_emotion=dominant,
                intensity=intensity
            ))
        
        return scores
    
    def _analyze_basic(self, segments: List[TextSegment]) -> List[EmotionScore]:
        """Analyze using basic lexicon (fallback)."""
        scores = []
        
        for i, segment in enumerate(segments):
            words = re.findall(r'\b\w+\b', segment.text.lower())
            emotion_counts = defaultdict(int)
            
            for word in words:
                for emotion, keywords in BASIC_EMOTION_LEXICON.items():
                    if word in keywords:
                        emotion_counts[emotion] += 1
            
            total = sum(emotion_counts.values())
            emotions = {}
            
            if total > 0:
                for emo, count in emotion_counts.items():
                    emotions[emo] = count / total
            else:
                emotions = {'neutral': 1.0}
            
            dominant = max(emotions, key=emotions.get)
            intensity = emotions[dominant]
            
            scores.append(EmotionScore(
                segment_id=i,
                emotions=emotions,
                dominant_emotion=dominant,
                intensity=intensity
            ))
        
        return scores


class HeatmapGenerator:
    """Generates emotional heatmap visualizations."""
    
    def __init__(self, segments: List[TextSegment], scores: List[EmotionScore]):
        self.segments = segments
        self.scores = scores
        self.emotions = self._get_all_emotions()
    
    def _get_all_emotions(self) -> List[str]:
        """Get all unique emotions from scores."""
        emotions = set()
        for score in self.scores:
            emotions.update(score.emotions.keys())
        # Sort by predefined order, then alphabetically
        predefined = list(EMOTION_COLORS.keys())
        sorted_emotions = []
        for emo in predefined:
            if emo in emotions:
                sorted_emotions.append(emo)
                emotions.discard(emo)
        sorted_emotions.extend(sorted(emotions))
        return sorted_emotions
    
    def generate(self, output_path: str, title: str = "Emotional Heatmap",
                style: str = 'heatmap'):
        """Generate and save the heatmap visualization."""
        if style == 'heatmap':
            self._generate_heatmap(output_path, title)
        elif style == 'timeline':
            self._generate_timeline(output_path, title)
        elif style == 'radar':
            self._generate_radar(output_path, title)
        elif style == 'stream':
            self._generate_stream(output_path, title)
        else:
            self._generate_heatmap(output_path, title)
    
    def _generate_heatmap(self, output_path: str, title: str):
        """Generate a traditional heatmap visualization."""
        if not self.scores or not self.emotions:
            print("Warning: No emotional data to visualize")
            return
        
        # Create emotion matrix
        matrix = np.zeros((len(self.emotions), len(self.scores)))
        
        for j, score in enumerate(self.scores):
            for i, emotion in enumerate(self.emotions):
                matrix[i, j] = score.emotions.get(emotion, 0)
        
        # Create figure
        fig_height = max(6, len(self.emotions) * 0.5)
        fig_width = max(12, len(self.scores) * 0.3)
        fig, ax = plt.subplots(figsize=(min(fig_width, 20), min(fig_height, 15)))
        
        # Create heatmap
        cmap = sns.color_palette("YlOrRd", as_cmap=True)
        sns.heatmap(matrix, ax=ax, cmap=cmap, 
                   xticklabels=range(1, len(self.scores) + 1),
                   yticklabels=self.emotions,
                   cbar_kws={'label': 'Intensity'})
        
        ax.set_xlabel('Segment Number')
        ax.set_ylabel('Emotion')
        ax.set_title(title)
        
        # Rotate x labels if many segments
        if len(self.scores) > 20:
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Heatmap saved to: {output_path}")
    
    def _generate_timeline(self, output_path: str, title: str):
        """Generate a timeline visualization of dominant emotions."""
        if not self.scores:
            print("Warning: No emotional data to visualize")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # Top: Stacked area chart of all emotions
        x = range(len(self.scores))
        emotion_data = {emo: [] for emo in self.emotions}
        
        for score in self.scores:
            for emo in self.emotions:
                emotion_data[emo].append(score.emotions.get(emo, 0))
        
        colors = [EMOTION_COLORS.get(emo, '#808080') for emo in self.emotions]
        ax1.stackplot(x, *[emotion_data[emo] for emo in self.emotions],
                     labels=self.emotions, colors=colors, alpha=0.8)
        
        ax1.set_xlim(0, len(self.scores) - 1)
        ax1.set_ylim(0, 1)
        ax1.set_ylabel('Emotion Intensity')
        ax1.set_title(title)
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
        
        # Bottom: Dominant emotion timeline
        dominant_emotions = [score.dominant_emotion for score in self.scores]
        unique_emotions = list(set(dominant_emotions))
        emotion_to_num = {emo: i for i, emo in enumerate(unique_emotions)}
        y_values = [emotion_to_num[emo] for emo in dominant_emotions]
        colors_scatter = [EMOTION_COLORS.get(emo, '#808080') for emo in dominant_emotions]
        
        ax2.scatter(x, y_values, c=colors_scatter, s=50, alpha=0.8)
        ax2.set_yticks(range(len(unique_emotions)))
        ax2.set_yticklabels(unique_emotions)
        ax2.set_xlabel('Segment Number')
        ax2.set_ylabel('Dominant Emotion')
        ax2.set_xlim(-0.5, len(self.scores) - 0.5)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Timeline saved to: {output_path}")
    
    def _generate_radar(self, output_path: str, title: str):
        """Generate a radar/spider chart of overall emotions."""
        if not self.scores or not self.emotions:
            print("Warning: No emotional data to visualize")
            return
        
        # Calculate average emotion scores
        avg_emotions = defaultdict(float)
        for score in self.scores:
            for emo, val in score.emotions.items():
                avg_emotions[emo] += val
        
        for emo in avg_emotions:
            avg_emotions[emo] /= len(self.scores)
        
        # Prepare data for radar chart
        emotions = list(avg_emotions.keys())
        values = list(avg_emotions.values())
        
        # Number of variables
        num_vars = len(emotions)
        
        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        ax.plot(angles, values, 'o-', linewidth=2, color='#FF6B6B')
        ax.fill(angles, values, alpha=0.25, color='#FF6B6B')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(emotions, size=10)
        ax.set_title(title, size=14, y=1.1)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Radar chart saved to: {output_path}")
    
    def _generate_stream(self, output_path: str, title: str):
        """Generate a streamgraph visualization."""
        if not self.scores:
            print("Warning: No emotional data to visualize")
            return
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        x = np.arange(len(self.scores))
        emotion_data = {}
        
        for emo in self.emotions:
            emotion_data[emo] = np.array([s.emotions.get(emo, 0) for s in self.scores])
        
        # Smooth the data
        from scipy.ndimage import gaussian_filter1d
        smoothed_data = {}
        for emo, data in emotion_data.items():
            try:
                smoothed_data[emo] = gaussian_filter1d(data, sigma=1)
            except:
                smoothed_data[emo] = data
        
        colors = [EMOTION_COLORS.get(emo, '#808080') for emo in self.emotions]
        
        ax.stackplot(x, *[smoothed_data[emo] for emo in self.emotions],
                    labels=self.emotions, colors=colors, baseline='wiggle', alpha=0.8)
        
        ax.set_xlim(0, len(self.scores) - 1)
        ax.set_xlabel('Segment Number')
        ax.set_ylabel('Emotion Flow')
        ax.set_title(title)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Streamgraph saved to: {output_path}")
    
    def generate_report(self, output_path: str):
        """Generate a JSON report of the emotional analysis."""
        report = {
            'summary': {
                'total_segments': len(self.segments),
                'emotions_detected': self.emotions,
            },
            'emotion_distribution': {},
            'segments': []
        }
        
        # Calculate overall emotion distribution
        total_emotions = defaultdict(float)
        for score in self.scores:
            for emo, val in score.emotions.items():
                total_emotions[emo] += val
        
        total = sum(total_emotions.values())
        if total > 0:
            report['emotion_distribution'] = {
                emo: round(val / total * 100, 2) 
                for emo, val in sorted(total_emotions.items(), key=lambda x: -x[1])
            }
        
        # Add segment details
        for segment, score in zip(self.segments, self.scores):
            report['segments'].append({
                'id': score.segment_id,
                'text': segment.text[:100] + '...' if len(segment.text) > 100 else segment.text,
                'timestamp': segment.timestamp,
                'speaker': segment.speaker,
                'dominant_emotion': score.dominant_emotion,
                'intensity': round(score.intensity, 3),
                'emotions': {k: round(v, 3) for k, v in score.emotions.items()}
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"Report saved to: {output_path}")
        return report


def main():
    parser = argparse.ArgumentParser(
        description='Generate emotional heatmap from transcription files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported file formats:
  Text:      .txt, .md
  Documents: .docx, .pdf, .rtf, .odt, .epub
  Subtitles: .srt, .vtt, .sbv
  Data:      .json, .csv
  Web:       .html, .htm

Examples:
  %(prog)s transcript.txt
  %(prog)s interview.docx -o heatmap.png --style timeline
  %(prog)s video.srt --transformer --report
  %(prog)s meeting.json -o analysis.png --style radar
        """
    )
    
    parser.add_argument('input_file', nargs='?', help='Input transcription file')
    parser.add_argument('-o', '--output', help='Output image file (default: <input>_heatmap.png)')
    parser.add_argument('--style', choices=['heatmap', 'timeline', 'radar', 'stream'],
                       default='heatmap', help='Visualization style (default: heatmap)')
    parser.add_argument('--transformer', action='store_true',
                       help='Use transformer model for better accuracy (requires more resources)')
    parser.add_argument('--model', help='Specific transformer model to use')
    parser.add_argument('--report', action='store_true',
                       help='Also generate a JSON report')
    parser.add_argument('--title', help='Custom title for the visualization')
    parser.add_argument('--segment-size', type=int, default=200,
                       help='Target segment size in characters (default: 200)')
    parser.add_argument('--list-formats', action='store_true',
                       help='List supported file formats and exit')
    
    args = parser.parse_args()
    
    if args.list_formats:
        print("Supported file formats:")
        for ext, fmt in sorted(TextExtractor.SUPPORTED_FORMATS.items()):
            print(f"  {ext:8} - {fmt}")
        sys.exit(0)
    
    # Validate input file
    if not args.input_file:
        parser.error("the following arguments are required: input_file")
    
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)
    
    # Set output path
    if args.output:
        output_path = args.output
    else:
        output_path = str(input_path.stem) + '_heatmap.png'
    
    # Set title
    title = args.title or f"Emotional Heatmap: {input_path.name}"
    
    print(f"Processing: {args.input_file}")
    print(f"Output: {output_path}")
    print(f"Style: {args.style}")
    
    try:
        # Extract text
        print("\n[1/3] Extracting text...")
        extractor = TextExtractor(args.input_file, segment_size=args.segment_size)
        segments = extractor.extract()
        print(f"  Extracted {len(segments)} segments")
        
        if not segments:
            print("Error: No text content found in file")
            sys.exit(1)
        
        # Analyze emotions
        print("\n[2/3] Analyzing emotions...")
        analyzer = EmotionAnalyzer(
            use_transformer=args.transformer,
            model_name=args.model
        )
        scores = analyzer.analyze(segments)
        print(f"  Analyzed {len(scores)} segments")
        
        # Generate visualization
        print("\n[3/3] Generating visualization...")
        generator = HeatmapGenerator(segments, scores)
        generator.generate(output_path, title=title, style=args.style)
        
        # Generate report if requested
        if args.report:
            report_path = str(Path(output_path).stem) + '_report.json'
            report = generator.generate_report(report_path)
            
            # Print summary
            print("\n" + "="*50)
            print("EMOTION SUMMARY")
            print("="*50)
            for emo, pct in report['emotion_distribution'].items():
                bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
                print(f"{emo:15} {bar} {pct:5.1f}%")
        
        print("\n✓ Processing complete!")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
