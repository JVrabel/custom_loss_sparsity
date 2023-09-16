"""
Contains functionality for creating PyTorch DataLoaders for 
LIBS benchmark classification dataset.
"""

import os
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from load_libs_data import load_contest_train_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from utils import resample_spectra_df

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
    
    pickle_file_path = "data/data.pkl"
    
    if os.path.exists(pickle_file_path):
        with open(pickle_file_path, 'rb') as f:
            data_dict = pickle.load(f)
        X = data_dict['X']
        y = data_dict['y']
        samples = data_dict['samples']
    else:
        X, y, samples = load_contest_train_dataset(train_dir, spectra_count)
        with open(pickle_file_path, 'wb') as f:
            pickle.dump({'X': X, 'y': y, 'samples': samples}, f)
        
    wavelengths = X.columns
    
    new_wave = np.arange(350, 700, 0.05)
    X_new = resample_spectra_df(X, wavelengths, new_wave)
    del X
    X_train, X_val, y_train, y_val = train_test_split(X_new, y, test_size=split_rate, random_state=random_st, stratify=samples, shuffle=True)
    del y, samples, X_new
    
    y_train = y_train-1
    y_val = y_val-1
    
    scaler = Normalizer(norm='max')
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    X_train = torch.from_numpy(X_train).float()
    X_val = torch.from_numpy(X_val).float()
    
    y_train = torch.from_numpy(np.array(y_train)).long()
    y_val = torch.from_numpy(np.array(y_val)).long()
    
    X_train = X_train.to(device)
    X_val = X_val.to(device)
    y_train = y_train.to(device)
    y_val = y_val.to(device)
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_val, y_val)
    
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, test_dataloader, y_train


