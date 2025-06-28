import matplotlib.pyplot as plt
import numpy as np


def plotStats(lengths):
    if not lengths or len(lengths) == 0:
        print("No data to plot.")
        return

    lengths = np.array(lengths)
    window_size = 10
    num_windows = len(lengths) // window_size
    averages = [
        lengths[i*window_size:(i+1)*window_size].mean()
        for i in range(num_windows)
    ]

    plt.figure(figsize=(10, 6))
    plt.plot(averages, marker='o', label='Average per 10 samples')
    max_length = lengths.max()
    plt.axhline(y=max_length, color='r', linestyle='--', label='Max length')
    plt.axhline(y=10, color='g', linestyle='--', label='Length=10')
    overall_avg = lengths.mean()
    percent = np.sum(lengths >= 10) / len(lengths) * 100
    stats_text = (
        f"Average length: {overall_avg:.2f}\n"
        f"Max length: {max_length}\n"
        f"Lengths ≥ 10: {percent:.2f}%"
    )

    plt.gca().text(
        0.95, 0.95, stats_text,
        verticalalignment='top',
        horizontalalignment='right',
        transform=plt.gca().transAxes,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
    )

    plt.legend()
    plt.xlabel('Window index')
    plt.ylabel('Average length')
    plt.title(f'Average Lengths per {window_size} samples')
    plt.gcf().canvas.mpl_connect(
        'key_press_event',
        lambda event: plt.close() if event.key == 'escape' else None
    )
    plt.show()
