import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

# --- Model & Data Parameters (derived from original TF cross_vali_recurrent_network_wifi_activity.py
# and cross_vali_data_convert_merge.py / cross_vali_input_data.py) ---
window_size = 500   # Time steps in one input sequence
n_input = 90      # Features per time step (30 subcarriers * 3 antennas)
n_steps = window_size # Renamed for clarity for LSTM input
n_hidden = 200    # Number of hidden units in the LSTM
n_classes = 7     # Number of activity classes

# Parameters for data loading/preprocessing, matching cross_vali_data_convert_merge.py defaults
threshold = 60 # Percentage threshold for activity detection
skip_rows_data_loading = 2 # SKIPROW from cross_vali_input_data.py to avoid memory error / subsample


# --- PyTorch Model Architecture ---
class WifiHARLSTM(nn.Module):
    def __init__(self, input_size=n_input, hidden_size=n_hidden, num_classes=n_classes, num_layers=1, batch_first=True):
        super(WifiHARLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        # Corresponds to tf.contrib.rnn.BasicLSTMCell and tf.nn.static_rnn
        # PyTorch's nn.LSTM expects input sequences as [batch, seq, features] if batch_first=True
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=self.batch_first)

        # Corresponds to the final tf.matmul (Dense layer)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: [batch_size, n_steps, n_input] (e.g., [200, 500, 90])
        
        # Initialize hidden and cell states for the LSTM
        # h0 and c0 shape: (num_layers * num_directions, batch_size, hidden_size)
        # Assuming num_directions=1 as it's a BasicLSTMCell equivalent
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        # out: (batch_size, seq_length, hidden_size) if batch_first=True
        # _ : (h_n, c_n)
        out, _ = self.lstm(x, (h0, c0))
        
        # We need the output from the last timestep to feed into the fully connected layer.
        # If batch_first=True, this is out[:, -1, :].
        out = self.fc(out[:, -1, :]) 
        
        return out


# --- PyTorch Data Loader Classes ---
class WifiActivityDataset(Dataset):
    def __init__(self, features=None, labels=None, fold_num=None, input_files_dir="input_files/", 
                 window_size=window_size, threshold=threshold, skip_rows=skip_rows_data_loading):
        
        self.window_size = window_size
        self.n_input = n_input # Ensure n_input is accessible here for reshaping

        # Determine if data is preloaded or needs to be loaded from file
        if features is not None and labels is not None:
            # Data is preloaded (e.g., from KFold split in the training script)
            self.features = features
            self.labels = labels
        elif fold_num is not None:
            # Load data from file for a specific fold (as in original script's logic)
            self.input_files_dir = input_files_dir
            self.threshold = threshold
            self.fold_num = fold_num # This fold_num implies a specific pre-generated file set

            xx_file = os.path.join(self.input_files_dir, f"xx_{self.window_size}_{self.threshold}_{self.fold_num}.csv")
            yy_file = os.path.join(self.input_files_dir, f"yy_{self.window_size}_{self.threshold}_{self.fold_num}.csv")

            if not os.path.exists(xx_file) or not os.path.exists(yy_file):
                raise FileNotFoundError(f"Data files for fold {self.fold_num} not found. "
                                        f"Please ensure you've run 'cross_vali_data_convert_merge.py' "
                                        f"and specify the correct fold number or check file paths.")

            # Load all data first, then apply subsampling as per original TF input_data.py
            xx_full = np.array(pd.read_csv(xx_file, header=None)).astype(np.float32)
            yy_full = np.array(pd.read_csv(yy_file, header=None)).astype(np.float32)

            # Apply the subsampling: "SKIPROW = 2" in original means xx = xx[::2,:]
            if skip_rows > 1:
                xx_full = xx_full[::skip_rows, :]
                yy_full = yy_full[::skip_rows, :]

            # Eliminate "NoActivity" data (labels with [2,0,0,0,0,0,0,0]) as per original TF script
            # In the original, it checks if column 0 of yy is 2.0 (representing NoActivity)
            no_activity_mask = ~np.all(yy_full[:, 0] == 2, axis=0) # True if NOT [2,0,0,...] at index 0
            self.features = xx_full[no_activity_mask]
            self.labels = yy_full[no_activity_mask]
            
            print(f"Loaded fold {self.fold_num}: Raw features shape {xx_full.shape}, Processed features shape {self.features.shape}, Labels shape {self.labels.shape}")

        else:
            raise ValueError("Either 'features' and 'labels' (for preloaded data) or 'fold_num' (for file loading) must be provided.")

        # Ensure features are reshaped for LSTM input [num_samples, window_size, n_input]
        # The data from CSVs comes as [num_samples, window_size * n_input] flattened
        if self.features.ndim == 2: # If it's still 2D after loading/masking
            self.features = self.features.reshape(-1, self.window_size, self.n_input)
        
        # Labels are already one-hot encoded but might need to be long for CrossEntropyLoss if not directly
        # using outputs, labels (assuming criterion(outputs, labels) where labels are one-hot float)
        # If using nn.CrossEntropyLoss with integer labels (standard):
        # self.labels = torch.argmax(torch.from_numpy(self.labels), dim=1).long()
        # For now, keep as float and let the training script handle `torch.max(labels.data, 1)`

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Convert NumPy arrays to PyTorch tensors
        sample_features = torch.from_numpy(self.features[idx])
        sample_labels = torch.from_numpy(self.labels[idx])
        
        return sample_features, sample_labels

# You can still import individual components from this file in your training script like:
# from wifi_har_pytorch_utils import WifiHARLSTM, WifiActivityDataset, n_input, n_hidden, n_classes, window_size, threshold