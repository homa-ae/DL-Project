from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "f1": f1_score(y_true, y_pred, average='macro', zero_division=0),
    }


def average_kfold_metrics(fold_metrics):
    # Initialize
    keys = fold_metrics[0].keys()
    averages = {key: 0.0 for key in keys}

    # Sum for each metrics
    for metrics in fold_metrics:
        for key in keys:
            averages[key] += metrics[key]

    # mean
    num_folds = len(fold_metrics)
    for key in keys:
        averages[key] /= num_folds

    return averages


def average_kfold_history(histories):

    avg_history = {}

    for key in histories[0].keys():
        all_folds_metric = [hist[key] for hist in histories]
        epochs = len(all_folds_metric[0])

        avg_history[key] = [
            sum(fold[i] for fold in all_folds_metric) / len(all_folds_metric)
            for i in range(epochs)
        ]

    return avg_history