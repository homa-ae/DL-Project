config = {
    "sample_rate": 16000,
    "n_mfcc": 40,
    "n_mels": 64,
    "feature_type": "mfcc",  # or "mel"
    "num_speakers": 10,
    "batch_size": 32,
    "num_epochs": 20,
    "learning_rate": 1e-3,
    "model_type": "cnn",     # or "rnn", "crnn"
    "k_folds": 5,
    "segment_duration": 3.0
}