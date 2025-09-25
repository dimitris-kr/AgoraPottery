import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchinfo

from utils import evaluate


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
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    losses = []
    # loss_weights = [1, 0.5]
    for X_batch, y_batch in loader:
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

    # Calculate Training Mean Loss
    train_loss = np.mean(losses)

    return model, optimizer, train_loss


def validate_epoch(model, loader, criterion, y_scaler):
    model.eval()
    losses = []
    y_true, y_pred = [], []
    # loss_weights = [1, 0.2]

    # Disable gradient computation
    with torch.no_grad():
        for X_batch, y_batch in loader:
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

            y_true += [y_batch]
            y_pred += [outputs]

    # Calculate Valication Mean Loss
    val_loss = np.mean(losses)

    # Concatenate Batches
    y_true, y_pred = torch.cat(y_true), torch.cat(y_pred)

    if model.chronology_target == "years":
        y_true = y_scaler.inverse_transform(y_true.cpu().numpy())
        y_pred = y_scaler.inverse_transform(y_pred.cpu().numpy())

    # Classification case -> choose class index with max value
    if model.chronology_target == "periods":
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.argmax(dim=1).cpu().numpy()

    return y_true, y_pred, val_loss


def report_epoch(epoch, train_loss, val_loss, scores):
    end = " | "
    epoch_txt = f"Epoch {epoch:03d}"
    print(epoch_txt, end=end)
    print(f"Train Loss: {train_loss:.4f}", end=end)
    print(f"Val. Loss: {val_loss:.4f}", end="\n")
    for d in range(len(scores)):
        print(len(epoch_txt) * " ", end=end)
        print(f"target {d}", end=end)
        for i, (metric, score) in enumerate(scores[d].items()):
            print(f"{metric}: {score:.4f}", end=end if i < len(scores[d]) - 1 else "\n")


def early_stopping(model, val_loss, best, patience_counter, patience):
    stop = False
    if val_loss < best[0]:
        best = (val_loss, model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            stop = True
            print("** Early Stopping **")

    return stop, best, patience_counter


def train(model,
          train_loader,
          val_loader,
          criterion,
          metrics,
          y_scaler,
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

    best = (np.inf, None)  # 0: best validation loss / 1: best model state
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # Training phase
        model, optimizer, train_loss = train_epoch(model, train_loader, criterion, optimizer)

        # Validation phase
        y_true, y_pred, val_loss = validate_epoch(model, val_loader, criterion, y_scaler)

        # Evaluate & Report Results
        scores = []
        for d in range(y_true.shape[1]):
            scores += [evaluate(y_true[:, d], y_pred[:, d], metrics)]
        report_epoch(epoch, train_loss, val_loss, scores)

        # Scheduler
        # If validation loss stops improving, lower learning rate automatically
        scheduler.step(val_loss)

        # Early Stopping
        # If validation loss doesn’t improve for X epochs (patience),
        # stop training early and restore the best model weights
        stop, best, patience_counter = early_stopping(model, val_loss, best, patience_counter, patience)
        if stop: break

    model.load_state_dict(best[1])
    return model
