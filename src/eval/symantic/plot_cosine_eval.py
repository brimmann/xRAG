import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

def plot_similarity_distributions():
    """
    Loads similarity scores, plots their distributions (histogram with KDE)
    in separate windows.
    """
    # --- 1. Configuration ---
    file1 = './results/pretrain/brimmann2_xgemma3_1b_v1/similarity_scores.json'
    file2 = './results/pretrain/Hannibal046_xrag_7b/similarity_scores.json'
    file3 = './results/pretrain/brimmann2_xgemma3_1b_v0/similarity_scores.json'
    scores_key = 'similarity_scores'

    files_to_plot = [
        (file1, os.path.basename(os.path.dirname(file1))),
        (file2, os.path.basename(os.path.dirname(file2))),
        (file3, os.path.basename(os.path.dirname(file3)))
    ]

    # --- 2. Helper Functions ---
    def load_scores(filepath, key):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            scores = [float(s) for s in data[key]]
            return scores
        except (FileNotFoundError, KeyError, json.JSONDecodeError, TypeError, ValueError) as e:
            print(f"Error processing file {filepath}: {e}")
            return None

    # --- 3. Plotting Setup ---
    sns.set_theme(style="whitegrid")

    # --- 4. Generate Plots ---
    for filepath, label in files_to_plot:
        scores = load_scores(filepath, scores_key)

        if scores is None:
            continue

        # Create a new figure for each plot
        plt.figure(figsize=(10, 6))
        ax = plt.gca()

        print(f"--- Generating plot for {label} ---")

        # Plot Histogram with KDE curve overlaid
        sns.histplot(scores, ax=ax, bins=30, stat="density",
                     alpha=0.6, label='Histogram')
        sns.kdeplot(scores, ax=ax, color='red', lw=2.5, label='Density Curve')
        ax.set_title(f'Distribution for {label}', fontsize=16)
        ax.set_xlabel('Similarity Score', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.legend()

    # --- 5. Final Layout Adjustments & Display ---
    plt.show()

if __name__ == '__main__':
    plot_similarity_distributions()