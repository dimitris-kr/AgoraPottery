import torch
import torch.nn as nn
import torchinfo

# MODEL CLASS
class PotteryChronologyPredictor(nn.Module):
    def __init__(self,
                 input_sizes,
                 output_size,
                 hidden_size,
                 activation=nn.ReLU,
                 dropout=0.3,
                 blocks=3,
                 hidden_size_pattern="decreasing",
                 chronology_target = "years"
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            self.model.add_module(f"dense{i+1}", nn.Linear(block_input_size, block_output_size))
            self.model.add_module(f"activation{i+1}", activation())
            self.model.add_module(f"dropout{i+1}", nn.Dropout(dropout))

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
                    batch_dim = 0,
                    device=self.device,
                    col_names=("input_size", "output_size", "num_params", "mult_adds")
                )
            )

# TRAIN

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    train_losses = []
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

        train_losses += [loss.item()]

    return model, optimizer, train_losses