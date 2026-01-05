import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchinfo
from sklearn.model_selection import ParameterGrid
from tqdm.std import tqdm


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


# PREDICT

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
    std_preds = preds.std(axis=0).squeeze(0)    # (n_targets,)

    # Inverse scaling if needed
    if y_scaler is not None:
        mean_preds = y_scaler.inverse_transform(mean_preds.reshape(1, -1)).squeeze(0)
        std_preds = std_preds * y_scaler.scale_

    return mean_preds, std_preds
