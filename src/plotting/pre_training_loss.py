import pandas as pd
import matplotlib.pyplot as plt

def plot_dev_ppl(csv_path='results/pretrain/training_loss/dev_ppl.csv'):
    """
    Plot development perplexity from CSV file.
    
    Args:
        csv_path: Path to the CSV file containing Step and dev_ppl data
    """
    # Read the CSV file with proper quote handling
    df = pd.read_csv(csv_path, quotechar='"', skipinitialspace=True)
    
    # Extract columns and ensure they're numeric
    # Strip any remaining quotes and convert to numeric
    steps = pd.to_numeric(df['Step'].astype(str).str.strip('"'))
    dev_ppl = pd.to_numeric(df['sample_pretrain - dev_ppl'].astype(str).str.strip('"'))
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(steps, dev_ppl, marker='o', linewidth=2, markersize=8)
    
    # Customize the plot
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Development Perplexity', fontsize=12)
    plt.title('Development Perplexity vs Training Steps', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on each point
    for x, y in zip(steps, dev_ppl):
        plt.annotate(f'{y:.2f}', 
                    xy=(x, y), 
                    xytext=(5, 5), 
                    textcoords='offset points',
                    fontsize=9)
    
    # Set y-axis to start from 0 for better visualization
    plt.ylim(bottom=0)
    
    # Tight layout for better appearance
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('plots/pretrain_training_loss/dev_ppl_fixed.png', dpi=100, bbox_inches='tight')
    
    # Show the plot
    plt.show()
    
    # Print summary statistics
    print(f"Initial perplexity: {dev_ppl.iloc[0]:.2f}")
    print(f"Final perplexity: {dev_ppl.iloc[-1]:.2f}")
    print(f"Improvement: {dev_ppl.iloc[0] - dev_ppl.iloc[-1]:.2f}")
    print(f"Percentage improvement: {((dev_ppl.iloc[0] - dev_ppl.iloc[-1]) / dev_ppl.iloc[0] * 100):.1f}%")

# Example usage:
if __name__ == "__main__":
    plot_dev_ppl('/home/brimmann/works/xRAG/results/pretrain/training_loss/dev_ppl.csv')