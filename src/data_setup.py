"""
Contains functionality for creating PyTorch DataLoaders for 
LIBS benchmark classification dataset.
"""

import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from load_libs_data import load_contest_train_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
import numpy as np

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str, 
    batch_size: int, 
    num_classes: int,
    device: torch.device,
    num_workers: int=NUM_WORKERS, 
    split_rate: float=0.5,
    random_st: int=102,
    spectra_count: int=50
):
    """Creates training and validation DataLoaders.
    ...
    """

    X, y, samples = load_contest_train_dataset(train_dir, spectra_count)
    wavelengths = X.columns

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=split_rate, random_state=random_st, stratify=samples, shuffle=True)
    del X, y, samples

    y_train = y_train-1
    y_val = y_val-1

    scaler =  Normalizer(norm='max')
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Convert data to torch tensors
    X_train = torch.from_numpy(X_train).float() # removed: Add extra dimension for channels
    X_val = torch.from_numpy(X_val).float()  # Add extra dimension for channels

    # Convert y_train and y_val to PyTorch tensor and adjust them to zero-based index
    y_train = torch.from_numpy(np.array(y_train)).long()
    y_val = torch.from_numpy(np.array(y_val)).long()
    # y_train_onehot = torch.nn.functional.one_hot(y_train, num_classes=12)
    # y_val_onehot = torch.nn.functional.one_hot(y_val, num_classes=12)



    # Move data to device
    X_train = X_train.to(device)
    X_val = X_val.to(device) 
    y_train = y_train.to(device)
    y_val = y_val.to(device)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_val, y_val)
    
    # y_train_onehot = y_train_onehot.to(device)
    # y_val_onehot = y_val_onehot.to(device)

    # train_dataset = TensorDataset(X_train, y_train_onehot)
    # test_dataset = TensorDataset(X_val, y_val_onehot)

    # Create DataLoaders
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, test_dataloader, y_train
