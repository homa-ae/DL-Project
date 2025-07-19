import torchaudio.transforms as T
from config import config

def get_feature_extractor():
    if config["feature_type"] == "mfcc":
        return T.MFCC(
            sample_rate=config["sample_rate"],
            n_mfcc=config["n_mfcc"],
            melkwargs={
                "n_fft": 400,
                "hop_length": 160,
                "n_mels": config["n_mels"],
            }
        )
    elif config["feature_type"] == "mel":
        return T.MelSpectrogram(
            sample_rate=config["sample_rate"],
            n_fft=400,
            hop_length=160,
            n_mels=config["n_mels"]
        )
    else:
        raise ValueError("Unknown feature type: " + config["feature_type"])