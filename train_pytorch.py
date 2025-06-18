import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
import os
import sklearn.metrics as sk_metrics # For confusion matrix

# Import your model and dataset classes
from pytorch_model import WifiHARLSTM, WifiActivityDataset, n_input, n_hidden, n_classes, window_size, threshold

# --- Training Parameters (from cross_vali_recurrent_network_wifi_activity.py) ---
learning_rate = 0.0001
training_epochs = 2000 # Matches original training_iters
batch_size = 200
display_step = 100 # How often to print training progress

# --- Output Folder ---
OUTPUT_FOLDER_PATTERN = "PyTorch_LR{0}_BATCHSIZE{1}_NHIDDEN{2}/"
output_folder = OUTPUT_FOLDER_PATTERN.format(learning_rate, batch_size, n_hidden)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- K-Fold Cross-Validation Setup (from original TF script) ---
# The original script performs 5-fold cross-validation.
# We'll mimic this by loading all data and splitting it using KFold.
# The original script loads separate xx_*.csv for each fold, then uses specific folds for train/vali.
# Here, we'll iterate through `fold_num` for the Dataset class to get the correct data split.
num_folds = 5 # As per the original script

# List to store results
cv_accuracies = []
confusion_matrix_sum = np.zeros((n_classes, n_classes), dtype=int)

print("Starting K-Fold Cross-Validation...")

for fold_num in range(num_folds):
    print(f"\n--- Starting Fold {fold_num + 1}/{num_folds} ---")

    # Load data for the current fold using your Dataset class
    # The original TF script handles train/validation splits by using specific CSVs per fold.
    # We will replicate this by treating `fold_num` as the validation fold
    # and loading others for training.
    
    # For a direct 5-fold cross-validation in PyTorch, it's often more standard
    # to load the *entire* dataset first, then split it.
    # However, since your TF script loads data specific to each fold,
    # let's assume `cross_vali_data_convert_merge.py` creates files for each fold,
    # and fold `i`'s data represents the pre-processed data for that specific validation split.

    # This means for `fold_num`, we load `xx_train_fold_num_*.csv` and `yy_train_fold_num_*.csv`
    # and `xx_vali_fold_num_*.csv` and `yy_vali_fold_num_*.csv`.
    # The provided `cross_vali_data_convert_merge.py` creates `xx_{WS}_{TH}_{FOLD}.csv`.
    # The `cross_vali_recurrent_network_wifi_activity.py` then uses `csv_import` from `cross_vali_input_data.py`.
    # It appears the `csv_import` function handles loading the correct train/validation splits.

    # Let's simplify and assume for each `fold_num` (0-4), `WifiActivityDataset(fold_num)` gives us
    # the training data *for that fold's training split*.
    # For true cross-validation, you'd usually load *all* data once, then use `KFold`
    # or ensure your `WifiActivityDataset` can represent the full dataset and `DataLoader` handles splitting.

    # Given the original script uses separate files for folds (though its splitting logic is complex):
    # We'll adapt to load the 'training' and 'validation' data corresponding to the current fold
    # based on the original TF script's `csv_import` which seems to define splits implicitly.
    
    # In the provided cross_vali_recurrent_network_wifi_activity.py, the loop runs for i in range(5)
    # and then calls csv_import(i, learning_rate, batch_size, n_hidden)
    # The csv_import returns (x_train, y_train, x_vali, y_vali) for *that specific fold*.
    # So, we need two datasets for each fold: one for training and one for validation.
    
    # To precisely match the original TF script's data splitting for cross-validation:
    # The original `csv_import` function (in cross_vali_input_data.py) for fold `i`:
    #   - Loads `xx_1000_60_{j}.csv` and `yy_1000_60_{j}.csv` for j from 0 to 4.
    #   - If `j == i` (current fold index), it's used for validation.
    #   - If `j != i`, it's used for training.
    # We need to replicate this logic here.

    all_features = []
    all_labels = []
    for j in range(num_folds):
        temp_dataset = WifiActivityDataset(fold_num=j, window_size=window_size, threshold=threshold)
        all_features.append(temp_dataset.features)
        all_labels.append(temp_dataset.labels)

    all_features_np = np.concatenate(all_features, axis=0)
    all_labels_np = np.concatenate(all_labels, axis=0)

    # Manual KFold split to precisely match original cross-validation folds if needed:
    # The original script does a fixed split where one fold is validation and others are training.
    # The provided `cross_vali_input_data.py` DataSet class also has `next_batch`
    # and performs `numpy.random.shuffle(perm)` per epoch.

    # Let's use scikit-learn's KFold to manage the indices
    kf = KFold(n_splits=num_folds, shuffle=False) # Important: shuffle=False for reproducibility with original splits

    for train_index, val_index in kf.split(all_features_np):
        if np.array_equal(val_index, np.where(np.arange(len(all_features_np)) == fold_num * len(all_features[0]))[0]):
             # This is a hacky way to find the specific fold being used for validation
             # if the data was just concatenated and then KF split.
             # The original `csv_import` logic is more direct.
             # For simpler understanding, let's assume `WifiActivityDataset` can be adapted to return specific train/val for a given fold.
             # However, given `cross_vali_data_convert_merge.py` makes `xx_X_Y_fold#.csv` and `cross_vali_input_data.py` loads by fold#,
             # it means each `fold_num` implicitly represents a specific split where one is validation and others train.

             # Simpler approach:
             # Train data for current fold: All data EXCEPT `fold_num`'s data
             train_features_list = [all_features[j] for j in range(num_folds) if j != fold_num]
             train_labels_list = [all_labels[j] for j in range(num_folds) if j != fold_num]
             train_features = np.concatenate(train_features_list, axis=0)
             train_labels = np.concatenate(train_labels_list, axis=0)

             # Validation data for current fold: Only `fold_num`'s data
             val_features = all_features[fold_num]
             val_labels = all_labels[fold_num]
             
             train_dataset = WifiActivityDataset(features=train_features, labels=train_labels, preloaded=True) # Adapt WifiActivityDataset __init__
             val_dataset = WifiActivityDataset(features=val_features, labels=val_labels, preloaded=True) # Adapt WifiActivityDataset __init__
             break # Exit the KFold loop once the correct fold is identified/processed.

    # Need to modify WifiActivityDataset __init__ to accept preloaded data instead of fold_num
    # Quick fix in WifiActivityDataset:
    # class WifiActivityDataset(Dataset):
    #     def __init__(self, features=None, labels=None, fold_num=None, input_files_dir="input_files/", window_size=500, threshold=60, skip_rows=2, preloaded=False):
    #         if preloaded:
    #             self.features = features.reshape(-1, window_size, n_input) # Ensure correct shape
    #             self.labels = labels
    #         else:
    #             # Original loading logic using fold_num
    #             # ... (your existing __init__ code) ...
    #             pass
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) # No shuffle for validation

    # --- Model, Loss, Optimizer ---
    model = WifiHARLSTM(n_input, n_hidden, n_classes).to(device)
    criterion = nn.CrossEntropyLoss() # For multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Adam optimizer, matching TF

    # --- Training and Validation Metrics ---
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(training_epochs):
        # --- Training Phase ---
        model.train() # Set model to training mode
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels) # labels should be one-hot encoded for this criterion with target type torch.float32

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0) # Accumulate loss weighted by batch size
            
            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            _, true_labels_idx = torch.max(labels.data, 1) # Convert one-hot to index
            total_train += labels.size(0)
            correct_train += (predicted == true_labels_idx).sum().item()

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_accuracy = correct_train / total_train
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_accuracy)

        # --- Validation Phase ---
        model.eval() # Set model to evaluation mode
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0
        all_val_preds = []
        all_val_true = []

        with torch.no_grad(): # Disable gradient calculations for validation
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs.data, 1)
                _, true_labels_idx = torch.max(labels.data, 1) # Convert one-hot to index
                total_val += labels.size(0)
                correct_val += (predicted == true_labels_idx).sum().item()

                all_val_preds.extend(predicted.cpu().numpy())
                all_val_true.extend(true_labels_idx.cpu().numpy())

        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        epoch_val_accuracy = correct_val / total_val
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_accuracy)

        # --- Display Progress ---
        if (epoch + 1) % display_step == 0 or epoch == training_epochs - 1:
            print(f"Epoch [{epoch + 1}/{training_epochs}], "
                  f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_accuracy:.4f}, "
                  f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_accuracy:.4f}")

    # --- End of Fold ---
    print(f"Fold {fold_num + 1} Training Finished!")
    cv_accuracies.append(epoch_val_accuracy) # Store final accuracy for this fold
    
    # Calculate and accumulate confusion matrix for this fold
    conf_matrix = sk_metrics.confusion_matrix(all_val_true, all_val_preds)
    confusion_matrix_sum += conf_matrix
    print(f"Confusion Matrix for Fold {fold_num + 1}:\n{conf_matrix}")

    # Save Accuracy curve for this fold
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.ylim([0, 1])
    plt.title(f"Accuracy Curve - Fold {fold_num + 1}")
    plt.savefig(os.path.join(output_folder, f"Accuracy_Fold{fold_num + 1}.png"), dpi=150)
    plt.close() # Close figure to prevent displaying in script execution

    # Save Loss curve for this fold
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.ylim([0, 2])
    plt.title(f"Loss Curve - Fold {fold_num + 1}")
    plt.savefig(os.path.join(output_folder, f"Loss_Fold{fold_num + 1}.png"), dpi=150)
    plt.close() # Close figure

    # Optionally save the model checkpoint for this fold
    torch.save(model.state_dict(), os.path.join(output_folder, f"model_fold{fold_num + 1}.pth"))


print("\n--- Cross-Validation Finished! ---")
print(f"Average Cross-Validation Accuracy: {np.mean(cv_accuracies):.4f} +/- {np.std(cv_accuracies):.4f}")
print("Total Confusion Matrix (summed across all folds):\n", confusion_matrix_sum)

# Save overall confusion matrix
np.save(os.path.join(output_folder, "total_confusion_matrix.npy"), confusion_matrix_sum)