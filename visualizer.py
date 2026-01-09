import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_heatmap(df: pd.DataFrame, output_path: str):
    """
    Generates and saves an emotional heatmap from the dataframe.
    """
    if df.empty:
        print("No data to plot.")
        return

    # Set up the plot style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(15, 10))
    
    # Prepare data for heatmap
    # We want emotions on Y axis, Time/Chunks on X axis
    
    # Columns to exclude from heatmap values
    exclude_cols = ['chunk_index', 'text_snippet']
    emotion_cols = [col for col in df.columns if col not in exclude_cols]
    
    heatmap_data = df[emotion_cols].T
    
    # Create the heatmap
    # Use a diverging colormap or a sequential one depending on preference.
    # 'YlGnBu' is good for 0-1 scores.
    ax = sns.heatmap(heatmap_data, cmap="YlGnBu", cbar_kws={'label': 'Emotion Score'}, 
                     annot=False, fmt=".2f", linewidths=.5)
    
    plt.title('Emotional Heatmap of Transcription', fontsize=16)
    plt.xlabel('Text Chunks (Time Progression)', fontsize=12)
    plt.ylabel('Emotion', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300)
    print(f"Heatmap saved to {output_path}")
    
    # Also create a line plot for temporal evolution
    plt.figure(figsize=(15, 8))
    for emotion in emotion_cols:
        plt.plot(df['chunk_index'], df[emotion], label=emotion, linewidth=2, marker='o', markersize=4)
        
    plt.title('Emotional Trajectory Over Time', fontsize=16)
    plt.xlabel('Text Chunks', fontsize=12)
    plt.ylabel('Emotion Score', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    line_plot_path = output_path.replace('.png', '_line.png')
    plt.savefig(line_plot_path, dpi=300)
    print(f"Line plot saved to {line_plot_path}")
