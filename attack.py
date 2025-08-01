import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from pytorch_model import WifiHARLSTM, n_input, n_hidden, n_classes

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_FEATURES_PATH = "input_files/all_features_full.npy"
DEFAULT_LABELS_PATH = "input_files/all_labels_full.npy"
DEFAULT_MODEL_PATH = "PyTorch_LR0.0001_BATCHSIZE200_NHIDDEN200/checkpoint_fold5_epoch52.pth"

DEFAULT_STDDEVS = [0, 5, 10, 15, 20, 25, 30]
DEFAULT_EPSILONS = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_data(features_path: str, labels_path: str) -> DataLoader:
    features = np.load(features_path)
    labels = np.load(labels_path)
    dataset = TensorDataset(
        torch.from_numpy(features).float(),
        torch.from_numpy(labels).float(),
    )
    return DataLoader(dataset, batch_size=200, shuffle=True)

# ---------------------------------------------------------------------------
# Gaussian Noise Attack
# ---------------------------------------------------------------------------
def add_gaussian_noise(data: torch.Tensor, stddev: float) -> torch.Tensor:
    noise = torch.randn_like(data) * stddev
    return data + noise, noise

def test_gaussian(model, device, loader, stddev):
    correct, total = 0, 0
    peak_ratios = []

    for data, target_onehot in loader:
        data, target_onehot = data.to(device), target_onehot.to(device)
        target = torch.argmax(target_onehot, dim=1)

        noisy_data, noise = add_gaussian_noise(data, stddev)
        output = model(noisy_data)
        pred = output.max(1)[1]

        # Compute peak amplitude ratio
        ratio = noise.abs().max() / data.abs().max()
        peak_ratios.append(ratio.item())

        correct += (pred == target).sum().item()
        total += target.size(0)

    final_acc = correct / total
    avg_ratio = np.mean(peak_ratios)
    print(f"Gaussian Std={stddev} | Peak Ratio={avg_ratio:.4f} | Accuracy={final_acc:.4f}")
    return final_acc, avg_ratio

# ---------------------------------------------------------------------------
# FGSM Attack
# ---------------------------------------------------------------------------
def fgsm_attack(data, epsilon, data_grad):
    return data + epsilon * data_grad.sign()

def test_fgsm(model, device, loader, epsilon):
    correct, total_attacked = 0, 0
    peak_ratios = []

    for data, target_onehot in loader:
        data, target_onehot = data.to(device), target_onehot.to(device)
        target = torch.argmax(target_onehot, dim=1)
        data.requires_grad = True

        output = model(data)
        init_pred = output.max(1)[1]

        mask = (init_pred == target)
        if not mask.any():
            continue

        data_to_attack = data[mask]
        target_to_attack = target[mask]

        data_to_attack.retain_grad()
        output_correct = model(data_to_attack)
        loss = F.cross_entropy(output_correct, target_to_attack)
        model.zero_grad()
        loss.backward()

        grad = data_to_attack.grad
        perturbed = fgsm_attack(data_to_attack, epsilon, grad)
        noise = perturbed - data_to_attack

        peak_ratios.append(noise.abs().max() / data_to_attack.abs().max())

        output_adv = model(perturbed)
        final_pred = output_adv.max(1)[1]

        correct += (final_pred == target_to_attack).sum().item()
        total_attacked += len(target_to_attack)

    final_acc = 0 if total_attacked == 0 else correct / total_attacked
    avg_ratio = np.mean([r.item() for r in peak_ratios]) if peak_ratios else 0
    print(f"FGSM Eps={epsilon} | Peak Ratio={avg_ratio:.4f} | Accuracy={final_acc:.4f}")
    return final_acc, avg_ratio

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Compare FGSM and Gaussian noise attacks with peak amplitude normalization")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--features", default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--labels", default=DEFAULT_LABELS_PATH)
    parser.add_argument("--stddevs", nargs="*", type=float, default=DEFAULT_STDDEVS)
    parser.add_argument("--epsilons", nargs="*", type=float, default=DEFAULT_EPSILONS)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"Using {device} device")

    loader = load_data(args.features, args.labels)
    model = WifiHARLSTM(n_input, n_hidden, n_classes).to(device)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state)
    model.eval()

    # Gaussian
    gaussian_acc, gaussian_x = [], []
    for std in args.stddevs:
        acc, peak_ratio = test_gaussian(model, device, loader, std)
        gaussian_acc.append(acc)
        gaussian_x.append(peak_ratio)

    # FGSM
    fgsm_acc, fgsm_x = [], []
    for eps in args.epsilons:
        acc, peak_ratio = test_fgsm(model, device, loader, eps)
        fgsm_acc.append(acc)
        fgsm_x.append(peak_ratio)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(gaussian_x, gaussian_acc, "*-", label="Gaussian Noise Attack")
    plt.plot(fgsm_x, fgsm_acc, "o-", label="FGSM Attack")
    plt.xlabel("Peak Perturbation / Peak Original Amplitude")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy vs Normalized Peak Perturbation")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
