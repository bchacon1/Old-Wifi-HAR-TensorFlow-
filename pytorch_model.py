import torch
import torch.nn as nn

# --- Model Parameters (from cross_vali_recurrent_network_wifi_activity.py) ---
window_size = 500
n_input = 90  # WiFi activity data input (30 subcarriers * 3 antennas)
n_steps = window_size # timesteps
n_hidden = 200 # hidden layer num of features
n_classes = 7 # WiFi activity total classes

class WifiHARLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1, batch_first=True):
        super(WifiHARLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first # PyTorch LSTM can handle batch as first dimension

        # Corresponds to tf.contrib.rnn.BasicLSTMCell and tf.nn.static_rnn
        # tf.nn.static_rnn processes input sequences. PyTorch's nn.LSTM is more direct.
        # Ensure batch_first=True because our data will be [batch_size, n_steps, n_input]
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=batch_first)

        # Corresponds to tf.matmul(outputs[-1], weights['out']) + biases['out']
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: [batch_size, n_steps, n_input] (e.g., [200, 500, 90])
        
        # Initialize hidden and cell states for the LSTM
        # (num_layers * num_directions, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        # If batch_first=True:
        #   out: (batch_size, seq_length, hidden_size)
        #   _ : (h_n, c_n) where h_n and c_n are (num_layers * num_directions, batch_size, hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        
        # We need the output from the last timestep to feed into the fully connected layer.
        # If batch_first=True, this is out[:, -1, :].
        # If batch_first=False, this would be out[-1, :, :].
        out = self.fc(out[:, -1, :]) 
        
        return out

# Example of how to instantiate the model (you'll use this in your training script)
# model = WifiHARLSTM(n_input, n_hidden, n_classes).to(device) # device will be 'cuda'
# Updated WifiActivityDataset in pytorch_model.py (or data_loader.py)
class WifiActivityDataset(Dataset):
    def __init__(self, features=None, labels=None, fold_num=None, input_files_dir="input_files/", 
                 window_size=500, threshold=60, skip_rows=2):
        
        # Determine if data is preloaded or needs to be loaded from file
        if features is not None and labels is not None:
            # Data is preloaded (e.g., from KFold split)
            self.features = features
            self.labels = labels
        elif fold_num is not None:
            # Load data from file for a specific fold (as in original script's logic)
            self.input_files_dir = input_files_dir
            self.window_size = window_size
            self.threshold = threshold
            self.fold_num = fold_num

            xx_file = os.path.join(self.input_files_dir, f"xx_{self.window_size}_{self.threshold}_{self.fold_num}.csv")
            yy_file = os.path.join(self.input_files_dir, f"yy_{self.window_size}_{self.threshold}_{self.fold_num}.csv")

            if not os.path.exists(xx_file) or not os.path.exists(yy_file):
                raise FileNotFoundError(f"Data files for fold {fold_num} not found. "
                                        f"Please ensure you've run 'cross_vali_data_convert_merge.py' "
                                        f"and specify the correct fold number.")

            xx_full = np.array(pd.read_csv(xx_file, header=None)).astype(np.float32)
            yy_full = np.array(pd.read_csv(yy_file, header=None)).astype(np.float32)

            if skip_rows > 1:
                xx_full = xx_full[::skip_rows, :]
                yy_full = yy_full[::skip_rows, :]

            no_activity_mask = ~np.all(yy_full[:, 0] == 2, axis=0)
            self.features = xx_full[no_activity_mask]
            self.labels = yy_full[no_activity_mask]
            
        else:
            raise ValueError("Either 'features' and 'labels' or 'fold_num' must be provided.")

        # Ensure features are reshaped for LSTM input
        self.features = self.features.reshape(-1, self.window_size, n_input)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        sample_features = torch.from_numpy(self.features[idx])
        sample_labels = torch.from_numpy(self.labels[idx])
        return sample_features, sample_labels
# --- Parameters (ensure consistency with cross_vali_data_convert_merge.py) ---
window_size = 500 # Must match the window_size used when generating xx_*.csv
threshold = 60 # Must match the threshold used when generating xx_*.csv

class WifiActivityDataset(Dataset):
    def __init__(self, fold_num, input_files_dir="input_files/", window_size=500, threshold=60, skip_rows=2):
        """
        Initializes the dataset by loading pre-processed CSV files.
        Args:
            fold_num (int): The cross-validation fold number to load (e.g., 0, 1, 2, 3, 4).
            input_files_dir (str): Directory where xx_*.csv and yy_*.csv are stored.
            window_size (int): The window size used during data generation.
            threshold (int): The threshold used during data generation.
            skip_rows (int): To avoid memory error, original script skips rows.
                             Matches SKIPROW in cross_vali_input_data.py.
        """
        self.input_files_dir = input_files_dir
        self.window_size = window_size
        self.threshold = threshold
        self.fold_num = fold_num

        xx_file = os.path.join(self.input_files_dir, f"xx_{self.window_size}_{self.threshold}_{self.fold_num}.csv")
        yy_file = os.path.join(self.input_files_dir, f"yy_{self.window_size}_{self.threshold}_{self.fold_num}.csv")

        if not os.path.exists(xx_file) or not os.path.exists(yy_file):
            raise FileNotFoundError(f"Data files for fold {fold_num} not found. "
                                    f"Please ensure you've run 'cross_vali_data_convert_merge.py' "
                                    f"and specify the correct fold number.")

        # Replicating the skiprows logic from cross_vali_input_data.py
        # This reduces the dataset size to avoid memory errors, similar to the original TF script.
        num_lines = sum(1 for line in open(xx_file))
        # Skip every `skip_rows` rows. Original script skips x if x % SKIPROW != 0.
        # This means it keeps rows 0, SKIPROW, 2*SKIPROW, etc.
        # If SKIPROW = 2, it keeps row 0, 2, 4, ...
        # pandas skiprows argument takes a list of row numbers to skip.
        # So if we want to skip rows where `x % SKIPROW != 0`, we keep rows where `x % SKIPROW == 0`.
        # The line `skip_idx = [x for x in range(1, num_lines) if x % SKIPROW !=0]`
        # in the original code means it reads row 0, then skips row 1, reads row 2, skips row 3, etc.
        # This is effectively taking every 2nd row starting from the first.
        # pandas can do this with `::2` on the dataframe after reading or more efficiently with skiprows.
        
        # A simpler way to replicate original behavior of `xx = xx[::2,:]` (taking every second sample)
        # Load all data first, then subsample. This is safer than `skiprows` for replication.
        xx_full = np.array(pd.read_csv(xx_file, header=None)).astype(np.float32)
        yy_full = np.array(pd.read_csv(yy_file, header=None)).astype(np.float32)

        # Apply the subsampling: "SKIPROW = 2 #Skip every 2 rows" -> xx = xx[::2,:]
        # This means we take the first row, then skip 1, take the next, etc.
        if skip_rows > 1:
            xx_full = xx_full[::skip_rows, :]
            yy_full = yy_full[::skip_rows, :]

        # Eliminate "NoActivity" data (labels with [2,0,0,0,0,0,0,0])
        # This is copied from cross_vali_input_data.py
        no_activity_mask = ~np.all(yy_full[:, 0] == 2, axis=0) # True if NOT [2,0,0,...] at index 0
        self.features = xx_full[no_activity_mask]
        self.labels = yy_full[no_activity_mask]

        # Reshape features to [num_samples, n_steps, n_input]
        # It was flattened during saving (xx = xx.reshape(len(xx),-1)) in cross_vali_data_convert_merge.py
        # Need to reshape back for LSTM input
        self.features = self.features.reshape(-1, self.window_size, n_input)
        
        print(f"Loaded fold {fold_num}: Features shape {self.features.shape}, Labels shape {self.labels.shape}")


    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Convert NumPy arrays to PyTorch tensors
        sample_features = torch.from_numpy(self.features[idx])
        sample_labels = torch.from_numpy(self.labels[idx])
        
        return sample_features, sample_labels

# Example of how to use this (in your training script):
# train_dataset = WifiActivityDataset(fold_num=0) # For KFold, you'll iterate 0 to 4
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)