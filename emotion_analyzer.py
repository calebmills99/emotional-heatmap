import torch
from transformers import pipeline
import pandas as pd
import numpy as np
from typing import List, Dict, Any

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """
    Splits text into chunks of approximately chunk_size words.
    """
    words = text.split()
    chunks = []
    
    if len(words) <= chunk_size:
        return [text]
        
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        
    return chunks

def analyze_emotions(text: str, model_name: str, chunk_size: int = 512) -> pd.DataFrame:
    """
    Analyzes emotions in the text using the specified model.
    Returns a pandas DataFrame with emotion scores for each chunk.
    """
    # Initialize the emotion classifier
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline("text-classification", model=model_name, return_all_scores=True, device=device)
    
    chunks = chunk_text(text, chunk_size)
    results = []
    
    print(f"Analyzing {len(chunks)} text chunks...")
    
    for i, chunk in enumerate(chunks):
        # Truncate if necessary (though our chunking should handle it mostly)
        # Using simple truncation for now; a better approach would use tokenizers
        try:
            # Most models have a limit of 512 tokens
            # We take a subset of characters to be safe, roughly 4 chars per token
            # But the pipeline handles truncation if we pass truncation=True
            prediction = classifier(chunk, truncation=True, max_length=512)
            
            # Format: [[{'label': 'joy', 'score': 0.9}, ...]]
            emotion_scores = {item['label']: item['score'] for item in prediction[0]}
            emotion_scores['chunk_index'] = i
            emotion_scores['text_snippet'] = chunk[:50] + "..." # Store snippet for reference
            results.append(emotion_scores)
            
        except Exception as e:
            print(f"Error analyzing chunk {i}: {e}")
            continue
            
    return pd.DataFrame(results)
