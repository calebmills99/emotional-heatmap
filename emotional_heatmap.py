import argparse
import sys
import os

from file_handlers import read_file_content
from emotion_analyzer import analyze_emotions
from visualizer import plot_heatmap

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate emotional heatmap from transcription file.')
    parser.add_argument('input_file', help='Path to the transcription file')
    parser.add_argument('--output', '-o', default='emotional_heatmap.png', help='Path to save the heatmap image')
    parser.add_argument('--chunk_size', '-c', type=int, default=512, help='Text chunk size for analysis')
    parser.add_argument('--model', '-m', default='j-hartmann/emotion-english-distilroberta-base', help='HuggingFace emotion model to use')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # 1. Check if file exists
    if not os.path.exists(args.input_file):
        print(f"Error: File not found: {args.input_file}")
        sys.exit(1)

    print(f"Processing file: {args.input_file}")
    
    # 2. Read file content
    try:
        content = read_file_content(args.input_file)
        if not content:
            print("Error: Could not read file content or file is empty.")
            sys.exit(1)
        if len(content.strip()) == 0:
             print("Error: File extracted content is empty.")
             sys.exit(1)

        print(f"File read successfully. Text length: {len(content)} characters.")

    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    # 3. Analyze emotions
    print("Analyzing emotions... (this might take a moment depending on text length and hardware)")
    try:
        emotions_df = analyze_emotions(content, args.model, args.chunk_size)
    except Exception as e:
        print(f"Error analyzing emotions: {e}")
        sys.exit(1)
        
    if emotions_df.empty:
        print("Error: No emotions analyzed.")
        sys.exit(1)

    # 4. Generate heatmap
    print(f"Generating heatmap and saving to {args.output}...")
    try:
        plot_heatmap(emotions_df, args.output)
    except Exception as e:
        print(f"Error generating heatmap: {e}")
        sys.exit(1)
    
    print("Process completed successfully.")

if __name__ == '__main__':
    main()
