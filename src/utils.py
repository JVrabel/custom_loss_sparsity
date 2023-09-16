"""
Contains various utility functions for PyTorch model training and saving.
"""
import torch
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="xxx.pth")
  """
  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)

  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)




def resample_spectra_df(df, original_wavelengths, new_wavelengths):
    """
    Resample the spectra in dataset X (as a DataFrame) to new wavelengths using linear interpolation.

    Parameters:
    - df: DataFrame, where each row is a spectrum.
    - original_wavelengths: 1D array of original wavelengths.
    - new_wavelengths: 1D array of new wavelengths.

    Returns:
    - df_resampled: DataFrame of resampled spectra.
    """

    # Convert DataFrame to NumPy array
    X = df.values

    num_spectra = X.shape[0]
    X_resampled = np.zeros((num_spectra, len(new_wavelengths)))

    for i in range(num_spectra):
        f = interp1d(original_wavelengths, X[i, :], kind='linear', fill_value='extrapolate')
        X_resampled[i, :] = f(new_wavelengths)

    # Convert back to DataFrame
    df_resampled = pd.DataFrame(X_resampled, columns=new_wavelengths)

    return df_resampled
