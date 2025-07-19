import torchaudio
from torch.utils.data import Dataset
from collections import defaultdict
import random
import os
from config import config
from features.extractors import get_feature_extractor
from utils import set_seed

class SpeakerDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def load_librispeech_subset(set="train-clean-100"):
    dataset = torchaudio.datasets.LIBRISPEECH(
        root="./data",
        url=set,
        download=not os.path.isdir(f"./data/LibriSpeech/{set}") # Load if already downloaded, else download set
    )
    return dataset

def prepare_dataset():
    set_seed(config["seed"])
    dataset = load_librispeech_subset("train-clean-100")
    extractor = get_feature_extractor()

    # Group audio files by speaker
    speaker_to_samples = defaultdict(list)
    for waveform, sample_rate, _, speaker_id, _, _ in dataset:
        if sample_rate != config["sample_rate"]:
            waveform = torchaudio.functional.resample(waveform, sample_rate, config["sample_rate"])
        speaker_to_samples[speaker_id].append(waveform)

    speakers = sorted(list(speaker_to_samples.keys()))
    selected_speakers = random.sample(speakers, config["num_speakers"])
    speaker_label = {spk: i for i, spk in enumerate(selected_speakers)}

    X, y = [], []
    segment_len = int(config["segment_duration"] * config["sample_rate"])
    for spk in selected_speakers:
        for waveform in speaker_to_samples[spk]:
            if waveform.size(1) < segment_len:
                continue
            # découpe en segments
            for i in range(0, waveform.size(1) - segment_len + 1, segment_len):
                segment = waveform[:, i:i+segment_len]
                feature = extractor(segment).squeeze(0)  # (freq, time)
                if config["feature_type"] == "mfcc":
                    feature = feature.unsqueeze(0)  # (1, freq, time)
                else:
                    feature = feature.unsqueeze(0)  # (1, freq, time)
                X.append(feature)
                y.append(speaker_label[spk])

    # Shuffle and split
    indices = list(range(len(X)))
    random.shuffle(indices)
    train_split = int(0.8 * len(X))
    val_split = int(0.9 * len(X))
    X_train = [X[i] for i in indices[:train_split]]
    y_train = [y[i] for i in indices[:train_split]]
    X_val = [X[i] for i in indices[train_split:val_split]]
    y_val = [y[i] for i in indices[train_split:val_split]]
    X_test = [X[i] for i in indices[val_split:]]
    y_test = [y[i] for i in indices[val_split:]]

    return SpeakerDataset(X_train, y_train), SpeakerDataset(X_val, y_val), SpeakerDataset(X_test, y_test)