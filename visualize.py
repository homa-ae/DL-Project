import matplotlib.pyplot as plt
from config import config

def visualize_feature(feature, label="", save_path=False, show=True):
    plt.figure(figsize=(10, 4))
    plt.imshow(feature.numpy(), origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(format="%+2.0f dB")

    title = ("MFCC" if config["feature_type"] == "mfcc" else "Mel Spectrogram") + (" " + label if label else label)
    plt.title(title)

    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()