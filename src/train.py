"""
Trains a PyTorch model using device-agnostic code.
"""

import os
import torch
import data_setup, engine, model_builder, utils
import win32com.client
# Add this at the top of your script
import argparse

# Argument parser setup
parser = argparse.ArgumentParser(description='Training script for PyTorch model.')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to run')
parser.add_argument('--reg_type', type=str, default='vanilla', choices=['vanilla', 'L1', 'sparseloc'], help='Type of regularization to apply')
parser.add_argument('--reg_lambda', type=float, default=0.1, help='Regularization strength')

args = parser.parse_args()

# Then use args to set your hyperparameters
NUM_EPOCHS = args.epochs
regularization_type = args.reg_type
reg_lambda = args.reg_lambda



# Setup hyperparameters
# NUM_EPOCHS = 300
# regularization_type = "L1" # "vanilla" "sparseloc"
# reg_lambda = 0.1
BATCH_SIZE = 128
INPUT_SHAPE = 7000  # Modify this based on your actual input vector length
OUTPUT_SHAPE = 12
HIDDEN_UNITS1 = 256  # Number of neurons in the first hidden layer
HIDDEN_UNITS2 = 128  # Number of neurons in the second hidden layer
LEARNING_RATE = 0.001


# Setup directories for data - modify these paths as needed
shell = win32com.client.Dispatch("WScript.Shell")
shortcut = shell.CreateShortCut('data/contest_TRAIN.h5.lnk')
train_dir = shortcut.Targetpath

# train_dir = "data/train"
# test_dir = "data/test"   # this should be val, and also used only if there is a specific dataset for valiadation data. 

# Setup target device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, classes = data_setup.create_dataloaders(
    train_dir=train_dir,
    batch_size=BATCH_SIZE,
    device = device,
    num_classes = OUTPUT_SHAPE
)

# Create model with help from model_builder.py
model = model_builder.SimpleMLP(
    input_shape=INPUT_SHAPE,
    hidden_units1=HIDDEN_UNITS1,
    hidden_units2=HIDDEN_UNITS2,
    output_shape=OUTPUT_SHAPE
).to(device)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# Start training with help from engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device,
             regularization_type = regularization_type,
             reg_lambda = reg_lambda)

# Save the model with help from utils.py
utils.save_model(model=model,
                 target_dir="models",
                 model_name="simple_mlp_model.pth")
