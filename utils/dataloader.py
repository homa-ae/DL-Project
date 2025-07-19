import torchaudio
from torch.utils.data import Dataset
from collections import defaultdict
import random
import os
from config import config
from features.extractors import get_feature_extractor
from utils.set_seed import set_seed

class SpeakerDataset(Dataset):
    """
    Custom PyTorch Dataset for speaker audio features and corresponding labels.

    Args:
        data (list): List of extracted audio features (tensors).
        labels (list): List of corresponding speaker labels (integers).
    """
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        """
        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve a single data sample and its label by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (feature_tensor, label) where feature_tensor is the extracted
            audio feature tensor and label is the speaker's class index.
        """
        return self.data[idx], self.labels[idx]

def load_librispeech_subset(subset="train-clean-100", root_name="data"):
    """
    Load a subset of the LibriSpeech dataset, downloading if necessary.

    Args:
        subset (str, optional): Name of the LibriSpeech subset to load.
            Defaults to "train-clean-100".
        root_name (str, optional): Root directory where data is stored.
            Defaults to "data".

    Returns:
        torchaudio.datasets.LIBRISPEECH: The loaded LibriSpeech dataset subset.
    """

    print(f"Loading of {subset} subset...")

    # Create root folder if needed
    os.makedirs(f"./{root_name}", exist_ok=True)

    dataset = torchaudio.datasets.LIBRISPEECH(
        root=f"./{root_name}",
        url=subset,
        download=not os.path.isdir(f"./{root_name}/LibriSpeech/{subset}") # Load if already downloaded, else download set
    )

    print(f"Loading of {subset} done.")
    return dataset

def prepare_dataset():
    """
    Prepare the speaker dataset by extracting features from LibriSpeech audio,
    segmenting audio clips, and splitting into train, validation, and test sets.

    Uses configuration parameters from config.py for reproducibility and feature extraction.

    Returns:
        tuple: Three SpeakerDataset instances for training, validation, and testing.
    """
    set_seed(config["seed"]) # Really important to keep the same speakers through every run
    dataset = load_librispeech_subset("train-clean-100")
    extractor = get_feature_extractor() # Load the extractor type regarding config.py

    # Group audio files by speaker
    speaker_to_samples = defaultdict(list)
    for waveform, sample_rate, _, speaker_id, _, _ in dataset:
        # Resample audio regarding config.py if needed
        if sample_rate != config["sample_rate"]:
            waveform = torchaudio.functional.resample(waveform, sample_rate, config["sample_rate"])
        speaker_to_samples[speaker_id].append(waveform)

    speakers = sorted(list(speaker_to_samples.keys()))

    # Select n speakers randomly
    selected_speakers = random.sample(speakers, config["num_speakers"])

    # Associate new labels to speakers (for 0 to n-1)
    speaker_label = {spk: i for i, spk in enumerate(selected_speakers)}

    X, y = [], [] # X contains audio features, y contains label
    segment_len = int(config["segment_duration"] * config["sample_rate"])
    for spk in selected_speakers:
        
        # Consider only the required audio duration
        max_total_len = int(config["max_length_per_speaker"] * config["sample_rate"])
        accumulated_len = 0

        for waveform in speaker_to_samples[spk]:
            if accumulated_len >= max_total_len:
                break
            if waveform.size(1) < segment_len:
                continue
            for i in range(0, waveform.size(1) - segment_len + 1, segment_len):
                if accumulated_len + segment_len > max_total_len:
                    break
                segment = waveform[:, i:i+segment_len]
                # Extract features regarding config.py (mel-spect or MFCC)
                feature = extractor(segment).squeeze(0)
                if config["feature_type"] == "mfcc":
                    feature = feature.unsqueeze(0)  # (1, freq, time)
                else:
                    feature = feature.unsqueeze(0)  # (1, freq, time)
                X.append(feature)
                y.append(speaker_label[spk])
                accumulated_len += segment_len

    # Shuffle and split
    indices = list(range(len(X)))
    random.shuffle(indices)
    train_split = int(config["train_vol"] * len(X))
    val_split = int(config["train_vol"]+config["val_vol"] * len(X))
    X_train = [X[i] for i in indices[:train_split]]
    y_train = [y[i] for i in indices[:train_split]]
    X_val = [X[i] for i in indices[train_split:val_split]]
    y_val = [y[i] for i in indices[train_split:val_split]]
    X_test = [X[i] for i in indices[val_split:]]
    y_test = [y[i] for i in indices[val_split:]]

    return SpeakerDataset(X_train, y_train), SpeakerDataset(X_val, y_val), SpeakerDataset(X_test, y_test)