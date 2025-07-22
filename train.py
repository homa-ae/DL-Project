import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import config
from utils.data_preparation import prepare_dataset
from models.cnn import CNN
from evaluate import compute_metrics


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    # Looping over tqdm for the loading bar
    for features, labels in tqdm(loader, desc="Train", leave=False):
        features = features.unsqueeze(1).to(device)  # add channel dim
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * features.size(0)
        preds = outputs.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_preds)
    return epoch_loss, metrics


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad(): # Validation set so no weight update and no gradient desent
        # Looping over tqdm for the loading bar
        for features, labels in tqdm(loader, desc="Val", leave=False):
            features = features.unsqueeze(1).to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * features.size(0)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_preds)
    return epoch_loss, metrics


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare datasets
    train_set, val_set, test_set = prepare_dataset()

    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False)

    # Initialize model
    # infer input shape: sample random batch
    sample_feature, _ = train_set[0]
    input_shape = sample_feature.unsqueeze(0).shape  # (1, n_mfcc or n_mels, time)

    model = CNN(input_shape, config['num_speakers']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    best_val_loss = float('inf')

    # Training loop
    for epoch in range(1, config['num_epochs'] + 1):
        train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = validate_one_epoch(model, val_loader, criterion, device)

        print(f"Epoch {epoch}/{config['num_epochs']}")
        print(f"  Train Loss: {train_loss:.4f} | Acc: {train_metrics['accuracy']:.4f} | "
              f"F1: {train_metrics['f1']:.4f} | Prec: {train_metrics['precision']:.4f} | "
              f"Reca: {train_metrics['recall']:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Acc: {val_metrics['accuracy']:.4f} | "
              f"F1: {val_metrics['f1']:.4f} | Prec: {val_metrics['precision']:.4f} | "
              f"Reca: {val_metrics['recall']:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_cnn_speaker.pth')
            print("  Best model saved.")

    print("Training completed.")


if __name__ == '__main__':
    main()
