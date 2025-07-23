import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import config
from utils.data_preparation import prepare_dataset
from models import get_model_class
from evaluate import compute_metrics


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for features, labels in loader:
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
        for features, labels in loader:
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

    return epoch_loss, metrics, all_labels, all_preds

def train_model(train_set, val_set, epoch_details=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs("models/bests", exist_ok=True)

    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False)

    sample_feature, _ = train_set[0]
    input_shape = sample_feature.unsqueeze(0).shape

    ModelClass = get_model_class(config['model_type'])
    model = ModelClass(input_shape, config['num_speakers']).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    best_val_loss = float('inf')

    # Initialize an history of metrics for further plots
    history = {
    "train_loss": [],
    "val_loss": [],
    "train_accuracy": [],
    "val_accuracy": [],
    "train_f1": [],
    "val_f1": [],
    "train_precision": [],
    "val_precision": [],
    "train_recall": [],
    "val_recall": [],
    }

    # Loop over progression bar
    for epoch in tqdm(range(1, config['num_epochs'] + 1), desc="Training", unit="epoch"):
        train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics, val_labels, val_preds  = validate_one_epoch(model, val_loader, criterion, device)

        if epoch_details:
            print(f"Epoch {epoch}/{config['num_epochs']}")
            print(f"  Train Loss: {train_loss:.4f} | Acc: {train_metrics['accuracy']:.4f} | "
                f"F1: {train_metrics['f1']:.4f} | Prec: {train_metrics['precision']:.4f} | "
                f"Reca: {train_metrics['recall']:.4f}")
            print(f"  Val   Loss: {val_loss:.4f} | Acc: {val_metrics['accuracy']:.4f} | "
                f"F1: {val_metrics['f1']:.4f} | Prec: {val_metrics['precision']:.4f} | "
                f"Reca: {val_metrics['recall']:.4f}")
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["train_f1"].append(train_metrics["f1"])
        history["val_f1"].append(val_metrics["f1"])
        history["train_precision"].append(train_metrics["precision"])
        history["val_precision"].append(val_metrics["precision"])
        history["train_recall"].append(train_metrics["recall"])
        history["val_recall"].append(val_metrics["recall"])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_metrics = val_metrics
            best_train_metrics = train_metrics
            best_val_labels = val_labels
            best_val_preds = val_preds
            best_model_state = model.state_dict()
            torch.save(best_model_state, f"models/bests/best_{config['model_type']}_speaker.pth")
            if epoch_details:
                print("Best model saved.")
    
    print("Training completed.")
    model.load_state_dict(best_model_state)
    return model, best_train_metrics, best_val_metrics, history, best_val_labels, best_val_preds