"""Run a Fast Gradient Sign Method (FGSM) attack on a trained Wifi HAR model.

This script mirrors the structure of the PyTorch FGSM tutorial but operates on
the Wifi HAR dataset.  It loads a saved LSTM model and evaluates how the model
performs when adversarial perturbations of varying magnitude are applied to the
input sequences.
"""

import argparse
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from pytorch_model import WifiHARLSTM, n_input, n_hidden, n_classes

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Default dataset paths.  These files are produced by
# ``cross_vali_data_convert_merge.py`` and contain the entire processed
# training set.
DEFAULT_FEATURES_PATH = "input_files/all_features_full.npy"
DEFAULT_LABELS_PATH = "input_files/all_labels_full.npy"
# Default trained model path. Adjust if your checkpoint resides elsewhere.
DEFAULT_MODEL_PATH = "model.pth"

# Epsilon values to evaluate.  "0" corresponds to no attack so the value can be
# used as a baseline accuracy.
DEFAULT_EPSILONS = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]


def load_data(features_path: str, labels_path: str) -> DataLoader:
    """Load features and labels stored as ``.npy`` files.

    Each sequence is returned individually so that gradients can be computed
    with respect to every input example during the attack.
    """

    features = np.load(features_path)
    labels = np.load(labels_path)

    dataset = TensorDataset(
        torch.from_numpy(features).float(),
        torch.from_numpy(labels).float(),
    )

    # Batch size of 1 matches the tutorial implementation where each gradient
    # step is taken with respect to a single example.
    return DataLoader(dataset, batch_size=1, shuffle=True)


def fgsm_attack(data: torch.Tensor, epsilon: float, data_grad: torch.Tensor) -> torch.Tensor:
    """Given an input tensor, create its adversarial counterpart using FGSM."""

    # ``data_grad`` contains the gradient of the loss w.r.t ``data``.  The sign
    # of this tensor indicates the direction in which the loss increases the
    # fastest for each element of the input.
    sign_data_grad = data_grad.sign()

    # Generate the perturbed example by adjusting the original data by a small
    # step (``epsilon``) in the direction that maximizes the loss.
    perturbed = data + epsilon * sign_data_grad

    # Depending on the dataset's range you might want to clamp the values here.
    # The Wifi HAR features are not normalized to [0,1], so we leave the values
    # as-is.
    return perturbed


def test(model: torch.nn.Module, device: torch.device, loader: DataLoader, epsilon: float) -> Tuple[float, List[np.ndarray]]:
    """Evaluate ``model`` accuracy when subjected to an FGSM attack of strength ``epsilon``."""

    correct = 0
    adv_examples: List[np.ndarray] = []

    # Loop over the entire dataset
    for data, target_onehot in loader:
        data, target_onehot = data.to(device), target_onehot.to(device)

        # Convert one-hot labels to integer class indices
        target = torch.argmax(target_onehot, dim=1)

        # Enable gradient computation with respect to the input
        data.requires_grad = True

        # Forward pass
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]

        # Only bother attacking if the model got the prediction correct
        if init_pred.item() != target.item():
            continue

        # Calculate the loss and compute gradients
        loss = F.cross_entropy(output, target)
        model.zero_grad()
        loss.backward()

        # Collect the gradient of the data
        data_grad = data.grad.data

        # Create the perturbed example
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed input
        output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1]

        if final_pred.item() == target.item():
            correct += 1
            if epsilon == 0 and len(adv_examples) < 5:
                adv_examples.append(perturbed_data.squeeze().detach().cpu().numpy())
        else:
            if len(adv_examples) < 5:
                adv_examples.append(perturbed_data.squeeze().detach().cpu().numpy())

    final_acc = correct / float(len(loader))
    print(
        f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(loader)} = {final_acc:.4f}"
    )

    return final_acc, adv_examples


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Wifi HAR model under an FGSM attack",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Path to the saved model .pth file",
    )
    parser.add_argument(
        "--features",
        default=DEFAULT_FEATURES_PATH,
        help="Numpy file containing preprocessed features",
    )
    parser.add_argument(
        "--labels",
        default=DEFAULT_LABELS_PATH,
        help="Numpy file containing one-hot encoded labels",
    )
    parser.add_argument(
        "--epsilons",
        nargs="*",
        type=float,
        default=DEFAULT_EPSILONS,
        help="List of epsilon values to evaluate",
    )

    args = parser.parse_args()

    # Device selection: use GPU if available for faster computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # Prepare data loader and restore the trained model
    loader = load_data(args.features, args.labels)
    model = WifiHARLSTM(n_input, n_hidden, n_classes).to(device)

    state = torch.load(args.model, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()

    accuracies = []
    # Run the attack for each epsilon value
    for eps in args.epsilons:
        acc, _ = test(model, device, loader, eps)
        accuracies.append(acc)

    # Plot accuracy vs epsilon
    plt.figure(figsize=(5, 5))
    plt.plot(args.epsilons, accuracies, "*-")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epsilon")
    plt.show()


if __name__ == "__main__":
    main()

