import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from config import config


class Visualizer:
    def __init__(self, show=True, save_dir=None):
        self.show = show
        self.save_dir = save_dir

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            if not os.path.exists(self.save_dir):
                print(f"[Visualizer] Failed to create directory: {self.save_dir}")
            else:
                print(f"[Visualizer] Saving plots to directory: {self.save_dir}")

    def _sanitize_filename(self, title):
        """Convert plot title to a safe filename."""
        return re.sub(r'[^\w\-_. ]', '_', title).replace(" ", "_").lower()

    def _finalize_plot(self, title=None):
        if self.save_dir and title:
            filename = self._sanitize_filename(title) + ".png"
            full_path = os.path.join(self.save_dir, filename)
            plt.savefig(full_path)
            print(f"[Visualizer] Plot saved as: {full_path}")
        if self.show:
            plt.show()
        plt.close()


    def visualize_features(self, feature, label=""):
        plt.figure(figsize=(10, 4))
        plt.imshow(feature.numpy(), origin='lower', aspect='auto', cmap='viridis')
        plt.colorbar(format="%+2.0f dB")

        title = ("MFCC" if config["feature_type"] == "mfcc" else "Mel Spectrogram")
        if label:
            title += f" {label}"
        plt.title(title)

        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.tight_layout()

        self._finalize_plot(title)

    def plot_training_history(self, history, metrics=["loss"], title_prefix=""):
        epochs = range(1, len(history["train_loss"]) + 1)

        for metric in metrics:
            train_metric = history.get(f"train_{metric}")
            val_metric = history.get(f"val_{metric}")

            if train_metric and val_metric:
                plt.figure(figsize=(8, 5))
                plt.plot(epochs, train_metric, label="Train")
                plt.plot(epochs, val_metric, label="Validation")
                plt.xlabel("Epoch")
                plt.ylabel(metric.capitalize())
                title = f"{title_prefix}{metric.capitalize()} over Epochs"
                plt.title(title)
                plt.legend()
                plt.grid(True)
                plt.tight_layout()

                self._finalize_plot(title)
