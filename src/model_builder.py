"""
Contains PyTorch model code to instantiate an MLP model.
"""
import torch
from torch import nn

class SimpleMLP(nn.Module):
    """Creates a simple MLP architecture.

    Args:
        input_shape: An integer indicating the size of the input vector.
        hidden_units1: An integer indicating the number of hidden units in the first hidden layer.
        hidden_units2: An integer indicating the number of hidden units in the second hidden layer.
        output_shape: An integer indicating the number of output units.
    """
    def __init__(self, input_shape: int, hidden_units1: int, hidden_units2: int, output_shape: int) -> None:
        super().__init__()
        
        # First hidden layer
        self.hidden_layer_1 = nn.Sequential(
            nn.Linear(input_shape, hidden_units1),
            nn.ReLU()
        )
        
        # Second hidden layer
        self.hidden_layer_2 = nn.Sequential(
            nn.Linear(hidden_units1, hidden_units2),
            nn.ReLU()
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_units2, output_shape)
    
    def forward(self, x: torch.Tensor):
        x = self.hidden_layer_1(x)
        x = self.hidden_layer_2(x)
        x = self.output_layer(x)
        return x
