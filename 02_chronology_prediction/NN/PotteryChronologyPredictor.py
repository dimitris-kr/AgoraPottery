import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchinfo
from sklearn.model_selection import ParameterGrid
from tqdm.std import tqdm

from utils import evaluate, fmt_time
import time

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


# TRAIN
def train_epoch(model, loader, criterion, optimizer, desc):
    model.train()
    losses = []
    # loss_weights = [1, 0.5]
    loop = tqdm(loader, desc=f"{desc}[Train]", leave=True)
    for X_batch, y_batch in loop:
        X_batch = [_X_batch.to(model.device) for _X_batch in X_batch]
        y_batch = y_batch.to(model.device)

        optimizer.zero_grad()

        # Forward Pass → Predictions
        outputs = model(X_batch)

        # Calculate Loss
        loss = criterion(outputs, y_batch)

        # loss_per_target = []
        # for d in range(y_batch.shape[1]):
        #     loss_per_target += [criterion(outputs[:, d], y_batch[:, d])]
        #
        # loss = None
        # for w, l in zip(loss_weights, loss_per_target):
        #     wl = w * l
        #     loss = loss + wl if loss else wl

        # Backpropagation → Calculate Gradients
        loss.backward()

        # Update Weights
        optimizer.step()

        # Save batch loss
        losses += [loss.item()]

        loop.set_postfix(loss=f"{np.mean(losses):.4f}")

    # Calculate Training Mean Loss
    train_loss = np.mean(losses)

    return model, optimizer, train_loss


def validate_epoch(model, loader, criterion, y_scaler, desc):
    model.eval()
    losses = []
    y_true, y_pred = [], []
    # loss_weights = [1, 0.2]

    # Disable gradient computation
    with torch.no_grad():
        loop = tqdm(loader, desc=f"{desc}[ Val ]", leave=True)
        for X_batch, y_batch in loop:
            X_batch = [_X_batch.to(model.device) for _X_batch in X_batch]
            y_batch = y_batch.to(model.device)

            # Forward Pass → Predictions
            outputs = model(X_batch)

            # Calculate Loss
            loss = criterion(outputs, y_batch)
            # loss_per_target = []
            # for d in range(y_batch.shape[1]):
            #     loss_per_target += [criterion(outputs[:, d], y_batch[:, d])]
            #
            # loss = None
            # for w, l in zip(loss_weights, loss_per_target):
            #     wl = w * l
            #     loss = loss + wl if loss else wl

            # Save batch loss
            losses += [loss.item()]

            loop.set_postfix(loss=f"{np.mean(losses):.4f}")

            y_true += [y_batch]
            y_pred += [outputs]

    # Calculate Valication Mean Loss
    val_loss = np.mean(losses)

    # Concatenate Batches
    y_true, y_pred = torch.cat(y_true), torch.cat(y_pred)

    # Classification case -> choose class index with max value
    if model.chronology_target == "periods":
        y_pred = y_pred.argmax(dim=1)

    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    if y_scaler:
        y_true = y_scaler.inverse_transform(y_true)
        y_pred = y_scaler.inverse_transform(y_pred)

    return y_true, y_pred, val_loss

def report_epoch(history, start_time, epoch_txt):
    execution_time = time.time() - start_time

    sep = " | "
    line = f"{epoch_txt}{sep}"
    line += f"exec_time: {fmt_time(execution_time)}{sep}"
    for key, values in history.items():
        if key == "scores":
            for t, scores in enumerate(values):
                line += f"target{t}_"
                for i, (metric, score) in enumerate(scores.items()):
                    if i > 0: break
                    line += f"{metric}: {score[-1]:.4f}"
                    line += sep if t < len(values) - 1 else ""
        elif key == "best_epoch":
            continue
        else:
            line += f"{key}: {values[-1]:.4f}{sep}"
    print(line)

def report_scores(scores, indentation):
    for d in range(len(scores)):
        line = f"{indentation}target{d}: ["

        for i, (metric, score) in enumerate(scores[d].items()):
            line += f"{metric}: {score:.4f}"
            line += ", " if i < len(scores[d]) - 1 else "]"

        print(line)

def report_final_model(history):
    epoch_idx = history["best_epoch"] - 1

    print(f"** Final Model:")
    for key, values in history.items():
        if key == "scores":
            for t, scores in enumerate(values):
                line = f"   target{t}: ["

                for i, (metric, score) in enumerate(scores.items()):
                    line += f"{metric}: {score[epoch_idx]:.4f}"
                    line += ", " if i < len(scores) - 1 else "]"

                print(line)
        elif key == "best_epoch":
            continue
        else:
            print(f"   {key}: {values[epoch_idx]:.4f}")


def early_stopping(model, val_loss, best, patience_counter, patience, epoch):
    stop = False
    if val_loss < best[0]:
        best = (val_loss, model.state_dict(), epoch)
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            stop = True
            print("** Early Stopping")
            print(f"** Restore Model State at Epoch {best[2]}")

    return stop, best, patience_counter


def train(model,
          train_loader,
          val_loader,
          criterion,
          metrics,
          y_scaler=None,
          epochs=50,
          lr=1e-3,
          weight_decay=1e-5,
          patience=5
          ):
    optimizer = optim.Adam(
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

    best = (np.inf, None, 0)  # 0: best validation loss / 1: best model state / 2: best epoch
    patience_counter = 0

    # History containers
    history = {
        "train_loss": [],
        "val_loss": [],
        "scores": [{metric: [] for metric in metrics} for _ in range(val_loader.dataset.__dim__()[1])]
    }

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        epoch_txt = f"Epoch {str(epoch).zfill(len(str(epochs)))}/{epochs} "
        indentation = " " * len(epoch_txt)

        # Training phase
        # train_loop = tqdm(train_loader, desc=f"{epoch_txt}[Train]", leave=True)
        model, optimizer, train_loss = train_epoch(model, train_loader, criterion, optimizer, epoch_txt)

        # Validation phase
        # val_loop = tqdm(val_loader, desc=f"{indentation}[Val]", leave=True)
        y_true, y_pred, val_loss = validate_epoch(model, val_loader, criterion, y_scaler, indentation)

        # Evaluate & Report Results
        if len(y_true.shape) > 1:
            scores = [evaluate(y_true[:, d], y_pred[:, d], metrics) for d in range(y_true.shape[1])]
        else:
            scores = [evaluate(y_true, y_pred, metrics)]

        # report_scores(scores, indentation)

        # Save history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        # history["scores"].append(scores)

        for t in range(len(scores)):
            for metric in metrics:
                history["scores"][t][metric].append(scores[t][metric])


        # Scheduler
        # If validation loss stops improving, lower learning rate automatically
        scheduler.step(val_loss)

        # Early Stopping
        # If validation loss doesn’t improve for X epochs (patience),
        # stop training early and restore the best model weights
        stop, best, patience_counter = early_stopping(model, val_loss, best, patience_counter, patience, epoch)

        # report_epoch(history, start_time, epoch_txt)
        if stop: break

    model.load_state_dict(best[1])

    history["best_epoch"] = best[2]
    report_final_model(history)

    return model, history


def tune(param_grid, X_dim, y_dim, device):
    for params in ParameterGrid(param_grid):
        print(f"\n🔎 Running with params: {params}")
    print(len(ParameterGrid(param_grid)))