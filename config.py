config = {
    "sample_rate": 16000,
    "n_mfcc": 40,
    "n_mels": 64,
    "feature_type": "mfcc", # or "mel"
    "num_speakers": 10,
    "batch_size": 32,
    "num_epochs": 20,
    "learning_rate": 1e-3,
    "model_type": "cnn",    # or "rnn", "crnn"
    "k_folds": 5,
    "train_vol": 0.6,       # train+val+test should be 1
    "val_vol": 0.2, 
    "test_vol": 0.2,
    "segment_duration": 3.0,
    "seed": 1234
}