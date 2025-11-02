import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.widgets import Button
from functools import partial

def plot_similarity_distributions():
    """
    Loads similarity scores, plots their distributions (histogram with KDE)
    in a single window, and provides a button to save each plot individually.
    """
    # --- 1. Configuration ---
    file1 = './results/pretrain/brimmann2_xgemma3_1b_v1/similarity_scores.json'
    file2 = './results/pretrain/Hannibal046_xrag_7b/similarity_scores.json'
    scores_key = 'similarity_scores'

    files_to_plot = [
        (file1, os.path.basename(os.path.dirname(file1))),
        (file2, os.path.basename(os.path.dirname(file2)))
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

    def save_subplot(fig, ax, filename, event=None):
        """Saves a single subplot to a file."""
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(filename, bbox_inches=extent.expanded(1.2, 1.3), dpi=200)
        print(f"Saved '{filename}'")

    # --- 3. Plotting Setup ---
    sns.set_theme(style="whitegrid")
    num_files = len(files_to_plot)
    # Create a single column of plots, one for each file
    fig, axes = plt.subplots(num_files, 1, figsize=(10, 6 * num_files))
    if num_files == 1:
        axes = [axes] # Ensure axes is always iterable

    buttons = [] # Store button widgets to keep them responsive

    # --- 4. Generate Plots and Buttons ---
    for i, (filepath, label) in enumerate(files_to_plot):
        scores = load_scores(filepath, scores_key)
        ax = axes[i]

        if scores is None:
            ax.set_visible(False)
            continue

        print(f"--- Generating plot for {label} ---")

        # Plot Histogram with KDE curve overlaid
        sns.histplot(scores, ax=ax, bins=30, stat="density",
                     alpha=0.6, label='Histogram')
        sns.kdeplot(scores, ax=ax, color='red', lw=2.5, label='Density Curve')
        ax.set_title(f'Distribution for {label}', fontsize=16)
        ax.set_xlabel('Similarity Score', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.legend()

        # --- Add a Save Button for the subplot ---
        pos = ax.get_position()
        button_ax = fig.add_axes([pos.x0 + pos.width * 0.35, pos.y0 - 0.12, pos.width * 0.3, 0.05])
        button = Button(button_ax, 'Save Plot')

        filename = f'./{label.replace(" ", "_")}_distribution.png'
        callback = partial(save_subplot, fig, ax, filename)
        button.on_clicked(callback)
        buttons.append(button)

    # --- 5. Final Layout Adjustments & Display ---
    fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, hspace=0.5)
    plt.show()

if __name__ == '__main__':
    plot_similarity_distributions()