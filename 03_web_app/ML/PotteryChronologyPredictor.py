import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchinfo
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error, accuracy_score, \
    precision_score, recall_score, f1_score

# CONSTANTS

activation_funcs = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
}

metrics_r = {
    "mae": mean_absolute_error,
    "rmse": mean_squared_error,
    "r2": r2_score,
    "medae": median_absolute_error,
}

metrics_c = {
    "accuracy": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
}

metric_params = {metric: {"average": "macro", "zero_division": 0} for metric in ["precision", "recall", "f1"]}


# DATASET

class PotteryDataset(Dataset):
    def __init__(self, X_list, y):
        """
        X_list: list of tensors, each [N, d] (can be 1 or more feature sets)
        y: tensor of targets [N] or [N, t] (t = number of targets)
        """
        self.X_list = X_list
        self.y = y

    def __len__(self):
        # Return number of samples in dataset
        return self.y.shape[0]

    def __getitem__(self, idx):
        # Return one sample (features and target) at position idx
        return [X[idx] for X in self.X_list], self.y[idx]


# MODEL CLASS
# Multilayer Perceptron
# Hidden Layer Block:
# Linear → Activation → Dropout

class PotteryChronologyPredictor(nn.Module):
    def __init__(self,
                 input_sizes,
                 output_size,
                 hidden_size,
                 device,
                 activation=nn.ReLU,
                 dropout=0.3,
                 blocks=3,
                 hidden_size_pattern="decreasing",
                 chronology_target="years"
                 ):

        super(PotteryChronologyPredictor, self).__init__()

        # chronology_target:
        # "years"    →  REGRESSION
        # "periods"  →  CLASSIFICATION
        self.chronology_target = chronology_target

        self.input_sizes = input_sizes
        self.output_size = output_size
        self.hidden_size = hidden_size

        # Set current device to GPU, if available
        self.device = device

        # PER-MODALITY ENCODERS
        # Input Size:  input_sizes[i]
        # Output Size: hidden_size
        self.encoders = nn.ModuleList([
            nn.Sequential(
                # HIDDEN LAYER BLOCK (Linear → Activation → Dropout)
                nn.Linear(input_size, hidden_size),
                activation(),
                nn.Dropout(dropout),
            ) for input_size in input_sizes
        ])

        # MAIN (Fusion network)
        # Input Size:  sum of output sizes of each encoder, which always equal to hidden_size
        # Output Size: output_size
        self.model = nn.Sequential()
        block_input_size = hidden_size * len(self.encoders)
        block_output_size = hidden_size

        for i in range(blocks):
            # HIDDEN LAYER BLOCK (Linear → Activation → Dropout)
            self.model.add_module(f"dense{i + 1}", nn.Linear(block_input_size, block_output_size))
            self.model.add_module(f"activation{i + 1}", activation())
            self.model.add_module(f"dropout{i + 1}", nn.Dropout(dropout))

            block_input_size = block_output_size
            if hidden_size_pattern == "decreasing" and block_output_size > 8:
                block_output_size //= 2

        # OUTPUT LAYER
        self.model.add_module("output", nn.Linear(block_input_size, output_size))

        # Save full model on current device (GPU)
        self.to(self.device)

    def forward(self, *inputs):
        """
        inputs can be passed as:
          (a) Single tensor: model(x)
          (b) Multiple tensors: model(x1, x2, ...)
          (c) List/tuple of tensors: model([x1, x2, ...])
        each tensor has shape [batch_size, input_size_i]
        """

        # for case (c) flatten inputs: from ([x1, x2, ...]) to (x1, x2, ...)
        if len(inputs) == 1 and isinstance(inputs[0], (list, tuple)):
            inputs = inputs[0]

        inputs = [X.to(self.device) for X in inputs]

        # Pass each modality through its encoder
        encoded_inputs = [encoder(X) for X, encoder in zip(inputs, self.encoders)]

        # Concatenate encoded representations
        X = torch.cat(encoded_inputs, dim=1)  # [batch_size, hidden_size * n_inputs]

        # Pass through fusion network
        y = self.model(X)

        return y

    def summary(self, style="torchinfo"):
        if style == "pytorch":
            print(self)
        elif style == "torchinfo":
            print(
                torchinfo.summary(
                    self,
                    input_size=[(input_size,) for input_size in self.input_sizes],
                    batch_dim=0,
                    device=self.device,
                    col_names=("input_size", "output_size", "num_params", "mult_adds")
                )
            )


# PREDICT (INFERENCE)

def predict_periods_single(model, X, y_encoder):
    with torch.no_grad():
        X = [_X.to(model.device) for _X in X]
        output = model(X)

    y_probs = torch.softmax(output, dim=-1)
    y_pred = y_probs.argmax(dim=-1).item()

    y_probs = y_probs.tolist()[0]

    y_pred = y_encoder.inverse_transform([y_pred])[0]
    y_probs = {y_encoder.inverse_transform([i])[0]: prob for i, prob in enumerate(y_probs)}

    return y_pred, y_probs


def predict_years_single(
        model,
        X,
        y_scaler=None,
        mc_samples: int = 50,
):
    """
    Monte Carlo Dropout regression prediction for ONE item

    Returns:
        mean_preds: np.ndarray shape (n_targets,)
        std_preds: np.ndarray shape (n_targets,)
    """

    model.eval()

    # move inputs to device
    X = [_X.to(model.device) for _X in X]

    preds = []

    for _ in range(mc_samples):
        # Enable dropout at inference
        model.train()

        with torch.no_grad():
            output = model(X)  # shape: (1, n_targets)
            preds.append(output.cpu().numpy())

    # Shape: (mc_samples, 1, n_targets)
    preds = np.stack(preds)

    # Mean & std over MC samples
    mean_preds = preds.mean(axis=0).squeeze(0)  # (n_targets,)
    std_preds = preds.std(axis=0).squeeze(0)  # (n_targets,)

    # Inverse scaling if needed
    if y_scaler is not None:
        mean_preds = y_scaler.inverse_transform(mean_preds.reshape(1, -1)).squeeze(0)
        std_preds = std_preds * y_scaler.scale_

    return mean_preds, std_preds


# TRAIN  (production version)

def _train_epoch(model, loader, criterion, optimizer):
    model.train()
    losses = []
    for X_batch, y_batch in loader:
        X_batch = [_X_batch.to(model.device) for _X_batch in X_batch]
        y_batch = y_batch.to(model.device)

        optimizer.zero_grad()

        # Forward Pass → Predictions
        outputs = model(X_batch)

        # Calculate Loss
        loss = criterion(outputs, y_batch)

        # Backpropagation → Calculate Gradients
        loss.backward()

        # Update Weights
        optimizer.step()

        # Save batch loss
        losses += [loss.item()]

    # Calculate Training Mean Loss
    train_loss = float(np.mean(losses))

    return model, optimizer, train_loss


def _validate_epoch(model, loader, criterion, y_scaler=None):
    model.eval()
    losses = []
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = [_X_batch.to(model.device) for _X_batch in X_batch]
            y_batch = y_batch.to(model.device)

            # Forward Pass → Predictions
            outputs = model(X_batch)

            # Calculate Loss
            loss = criterion(outputs, y_batch)

            # Save batch loss
            losses += [loss.item()]

            y_true_all += [y_batch]
            y_pred_all += [outputs]

    # Calculate Validation Mean Loss
    val_loss = float(np.mean(losses))

    # Concatenate Batches
    y_true = torch.cat(y_true_all)
    y_pred = torch.cat(y_pred_all)

    # Classification case -> choose class index with max value
    if model.chronology_target == "periods":
        y_pred = y_pred.argmax(dim=1)

    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    if y_scaler:
        y_true = y_scaler.inverse_transform(y_true)
        y_pred = y_scaler.inverse_transform(y_pred)

    return y_true, y_pred, val_loss


def _evaluate(y_true, y_pred, metrics):
    scores = {}
    if len(y_true.shape) == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    for metric, get_metric_score in metrics.items():
        params = metric_params.get(metric, {})
        scores[metric] = []

        for d in range(y_true.shape[1]):
            metric_score = get_metric_score(y_true[:, d], y_pred[:, d], **params)

            if metric == "rmse": metric_score = np.sqrt(metric_score)
            scores[metric] += [float(metric_score)]
    return scores


def train(
        model: PotteryChronologyPredictor,
        train_loader: DataLoader,
        val_loader: DataLoader,
        y_scaler=None,  # StandardScaler (regression only)
        epochs: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        patience: int = 10,
):
    start_time = time.time()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    best = {
        "state": {},
        "train_loss": float("inf"),
        "val_loss": float("inf"),
        "scores": None,
    }

    patience_counter = 0

    if model.chronology_target == "periods":
        criterion = nn.CrossEntropyLoss()
        metrics = metrics_c
    elif model.chronology_target == "years":
        criterion = nn.MSELoss()
        metrics = metrics_r
    else:
        return

    for epoch in range(1, epochs + 1):
        # Training phase
        model, optimizer, train_loss = _train_epoch(model, train_loader, criterion, optimizer)

        # Validation phase
        y_true, y_pred, val_loss = _validate_epoch(model, val_loader, criterion, y_scaler)

        # Scheduler
        # If validation loss stops improving, lower learning rate automatically
        scheduler.step(val_loss)

        print(f"Epoch {epoch}/{epochs}: train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}")

        # Early Stopping
        # If validation loss doesn’t improve for X epochs (patience),
        # stop training early and restore the best model weights
        if val_loss < best["val_loss"]:
            best = {
                "state": {k: v.clone() for k, v in model.state_dict().items()},
                "train_loss": train_loss,
                "val_loss": val_loss,
                "scores": _evaluate(y_true, y_pred, metrics),
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best["state"])

    metadata = {
        "train_loss": best["train_loss"],
        "val_loss": best["val_loss"],
        "scores": best["scores"],
        "time": time.time() - start_time,
    }

    return model, metadata
