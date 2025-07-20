import torchaudio
import torch.nn
from torch.utils.data import Dataset
from collections import defaultdict
import random
import os
import glob
from config import config
from features.extractors import get_feature_extractor
from utils.set_seed import set_seed

class SpeakerDataset(Dataset):
    def __init__(self, file_list, speaker_to_idx):
        self.file_list = file_list  # [(file_path, speaker_id)]
        self.speaker_to_idx = speaker_to_idx
        self.extractor = get_feature_extractor()
        self.segment_len = int(config["segment_duration"] * config["sample_rate"])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path, speaker_id = self.file_list[idx]
        waveform, sr = torchaudio.load(path)
        if sr != config["sample_rate"]:
            waveform = torchaudio.functional.resample(waveform, sr, config["sample_rate"])

        # If audio is to short, add padding
        if waveform.size(1) < self.segment_len:
            padding = self.segment_len - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        # Get random segment if audio > segment_len (data augmentation)
        if waveform.size(1) > self.segment_len:
            max_start = waveform.size(1) - self.segment_len
            start = random.randint(0, max_start)
            segment = waveform[:, start:start+self.segment_len]
        else:
            segment = waveform

        feature = self.extractor(segment).squeeze(0)
        feature = feature.unsqueeze(0)  # (1, freq, time)
        return feature, self.speaker_to_idx[speaker_id]

def get_speaker_dirs(root_dir):
    pattern = os.path.join(root_dir, "LibriSpeech", "train-clean-100", "*", "*", "*.flac")
    return glob.glob(pattern)

def prepare_dataset(subset="train-clean-100", root_dir="data"):
    set_seed(config["seed"])

    # Dowload dataset if needed
    os.makedirs(f"./{root_dir}", exist_ok=True)
    torchaudio.datasets.LIBRISPEECH(root=f"./{root_dir}", url=subset, download=True)

    # Get every audio file
    all_files = get_speaker_dirs(f"./{root_dir}")

    # Link .flac file by speaker ID
    speaker_files = defaultdict(list)
    for file_path in all_files:
        parts = file_path.split(os.sep)
        speaker_id = int(parts[-3])
        speaker_files[speaker_id].append(file_path)

    all_speakers = list(speaker_files.keys())

    # Create a subset of n speaker regarding config.py
    selected_speakers = random.sample(all_speakers, config["num_speakers"])
    speaker_to_idx = {spk: idx for idx, spk in enumerate(selected_speakers)}

    selected_files = []

    # For every speaker of the previously defined speaker, limit to n second
    max_len_per_spk = config["max_length_per_speaker"]  # in second regarding config.py
    for spk in selected_speakers:
        total_len = 0.0
        for f in speaker_files[spk]:
            info = torchaudio.info(f)
            duration = info.num_frames / info.sample_rate
            if total_len + duration > max_len_per_spk:
                break
            selected_files.append((f, spk))
            total_len += duration

    # Shuffle & split
    random.shuffle(selected_files)
    n_total = len(selected_files)
    n_train = int(n_total * config["train_vol"])
    n_val = int(n_total * config["val_vol"])
    train_files = selected_files[:n_train]
    val_files = selected_files[n_train:n_train + n_val]
    test_files = selected_files[n_train + n_val:]

    return (
        SpeakerDataset(train_files, speaker_to_idx),
        SpeakerDataset(val_files, speaker_to_idx),
        SpeakerDataset(test_files, speaker_to_idx)
    )
