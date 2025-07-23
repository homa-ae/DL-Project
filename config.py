config = {
    "sample_rate": 16000,
    "n_mfcc": 40,
    "n_mels": 64,
    "feature_type": "mel", # "mfcc" or "mel"
    "audio_normalize": True, 
    "feature_normalize": False,
    "num_speakers": 10,
    "max_length_per_speaker": 60.0,  # max duration for every speaker in second
    "batch_size": 32,
    "num_epochs": 20,
    "learning_rate": 1e-3,
    "model_type": "resnet",    # or "rnn", "crnn"
    "k_folds": 5,
    "train_vol": 0.6,       # train+val+test should be 1
    "val_vol": 0.2, 
    "test_vol": 0.2,
    "segment_duration": 3.0,
    "seed": 1234
}