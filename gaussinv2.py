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
DEFAULT_STDDEVS = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

def load_data(features_path: str, labels_path: str) -> DataLoader:
    features = np.load(features_path)
    labels = np.load(labels_path)
    dataset = TensorDataset(
        torch.from_numpy(features).float(),
        torch.from_numpy(labels).float(),
    )
    return DataLoader(dataset, batch_size=200, shuffle=True)

def add_gaussian_noise(data: torch.Tensor, stddev: float) -> torch.Tensor:
    noise = torch.randn_like(data) * stddev
    print(f"Injected full-sequence noise: std={noise.std().item():.4f}, max={noise.max().item():.4f}")
    return data + noise

def add_transition_noise(data: torch.Tensor, stddev: float, transition_portion: float = 0.2) -> torch.Tensor:
    # Inject noise only in the last portion of each input sequence
    perturbed = data.clone()
    B, T, F = data.shape
    start_idx = int(T * (1 - transition_portion))

    for i in range(B):
        noise = torch.randn_like(data[i]) * stddev
        perturbed[i, start_idx:] += noise[start_idx:]
        print(f"Injected transition noise: std={noise.std().item():.4f}, max={noise.max().item():.4f}")

    return perturbed

def test(model: torch.nn.Module, device: torch.device, loader: DataLoader, stddev: float):
    correct = 0
    total = 0
    adv_examples = []

    for data, target_onehot in loader:
        data, target_onehot = data.to(device), target_onehot.to(device)
        target = torch.argmax(target_onehot, dim=1)

        noisy_data = add_gaussian_noise(data, stddev)
        output = model(noisy_data)
        pred = output.max(1)[1]

        correct += (pred == target).sum().item()
        total += target.size(0)

        if stddev == 0 and len(adv_examples) < 5:
            adv_examples.extend(noisy_data[:5].detach().cpu().numpy())

    final_acc = correct / total
    print(f"Stddev: {stddev}\tTest Accuracy = {correct} / {total} = {final_acc:.4f}")
    return final_acc, adv_examples

def test_transition_noise(model: torch.nn.Module, device: torch.device, loader: DataLoader, stddev: float):
    correct = 0
    total = 0

    for data, target_onehot in loader:
        data, target_onehot = data.to(device), target_onehot.to(device)
        target = torch.argmax(target_onehot, dim=1)

        noisy_data = add_transition_noise(data, stddev)
        output = model(noisy_data)
        pred = output.max(1)[1]

        correct += (pred == target).sum().item()
        total += target.size(0)

    final_acc = correct / total
    print(f"[Transition Noise] Stddev: {stddev:.2f} | Accuracy = {correct} / {total} = {final_acc:.4f}")
    return final_acc

def main():
    parser = argparse.ArgumentParser(description="Gaussian noise attack on Wifi HAR model")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to trained model .pth file")
    parser.add_argument("--features", default=DEFAULT_FEATURES_PATH, help="Numpy features file")
    parser.add_argument("--labels", default=DEFAULT_LABELS_PATH, help="Numpy labels file")
    parser.add_argument("--stddevs", nargs="*", type=float, default=DEFAULT_STDDEVS, help="List of stddev values")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"Using {device} device")

    loader = load_data(args.features, args.labels)
    model = WifiHARLSTM(n_input, n_hidden, n_classes).to(device)
    state = torch.load(args.model, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()

    accuracies = []
    for std in args.stddevs:
        acc, _ = test(model, device, loader, std)
        accuracies.append(acc)

    print("\n=== Transition-only Gaussian Noise Attack ===")
    for std in args.stddevs:
        test_transition_noise(model, device, loader, std)

    plt.figure(figsize=(5, 5))
    plt.plot(args.stddevs, accuracies, "*-", label="Full-sequence Noise")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xlabel("Stddev")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Gaussian Noise Stddev")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
