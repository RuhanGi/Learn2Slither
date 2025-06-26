import matplotlib.pyplot as plt
import numpy as np

def plotStats(lengths):
    if not lengths:
        print("No data to plot.")
        return

    lengths = np.array(lengths)
    avg_len = lengths.mean()
    max_len = lengths.max()
    pct_above_10 = (lengths >= 10).sum() / len(lengths) * 100

    plt.figure(figsize=(10, 6))
    plt.plot(lengths, linestyle='None', label='Length per Episode', color='blue', marker='o')
    plt.axhline(avg_len, color='orange', linestyle='--', label=f'Average = {avg_len:.2f}')
    plt.axhline(max_len, color='green', linestyle='-.', label=f'Max = {max_len}')
    
    # Add percentage text
    plt.text(len(lengths)*0.7, max_len*0.9,
             f'≥10 Length Episodes: {pct_above_10:.1f}%',
             fontsize=12, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round'))

    plt.title("Snake Length per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Snake Length")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.gcf().canvas.mpl_connect('key_press_event', lambda event: plt.close() if event.key == 'escape' else None)
    plt.show()
