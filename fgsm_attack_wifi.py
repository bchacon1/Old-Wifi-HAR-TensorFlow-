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
DEFAULT_EPSILONS = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

def load_data(features_path: str, labels_path: str) -> DataLoader:
    features = np.load(features_path)
    labels = np.load(labels_path)
    dataset = TensorDataset(
        torch.from_numpy(features).float(),
        torch.from_numpy(labels).float(),
    )
    return DataLoader(dataset, batch_size=200, shuffle=True)

def fgsm_attack(data: torch.Tensor, epsilon: float, data_grad: torch.Tensor) -> torch.Tensor:
    sign_data_grad = data_grad.sign()
    perturbed = data + epsilon * sign_data_grad
    return perturbed
"""
def test(model: torch.nn.Module, device: torch.device, loader: DataLoader, epsilon: float):
    correct = 0
    total_attacked = 0
    adv_examples = []

    print(f"\n--- FGSM Attack: Epsilon = {epsilon} ---")

    for i, (data, target_onehot) in enumerate(loader):
        data, target_onehot = data.to(device), target_onehot.to(device)
        target = torch.argmax(target_onehot, dim=1)  # shape: [batch_size]

        data.requires_grad = True
        output = model(data)  # shape: [batch_size, num_classes]
        init_pred = output.max(1)[1]  # shape: [batch_size]

        # Only attack samples that were correctly predicted
        mask = (init_pred == target)
        if not mask.any():
            continue  # skip this batch if no correct predictions

        # Filter data for correctly predicted examples
        data_to_attack = data[mask]
        target_to_attack = target[mask]

        # Compute gradients on filtered subset
        loss = F.cross_entropy(model(data_to_attack), target_to_attack)
        model.zero_grad()
        data_to_attack.retain_grad()
        loss.backward()
        data_grad = data_to_attack.grad.data

        # Generate adversarial examples
        perturbed_data = fgsm_attack(data_to_attack, epsilon, data_grad)

        # Re-classify perturbed data
        output_adv = model(perturbed_data)
        final_pred = output_adv.max(1)[1]

        # Count how many were still correct
        correct += (final_pred == target_to_attack).sum().item()
        total_attacked += len(target_to_attack)

        # Save a few adversarial examples
        if epsilon == 0 and len(adv_examples) < 5:
            adv_examples.extend(perturbed_data[:5].detach().cpu().numpy())

    if total_attacked == 0:
        print("⚠️  No correctly predicted samples to attack at this epsilon.")
        final_acc = 0.0
    else:
        final_acc = correct / total_attacked

    print(f"Epsilon: {epsilon} | Accuracy: {correct} / {total_attacked} = {final_acc:.4f}")
    return final_acc, adv_examples
"""
def test(model: torch.nn.Module, device: torch.device, loader: DataLoader, epsilon: float):
    correct = 0
    total_attacked = 0
    adv_examples = []

    print(f"\n--- FGSM Attack: Epsilon = {epsilon} ---")

    for i, (data, target_onehot) in enumerate(loader):
        data, target_onehot = data.to(device), target_onehot.to(device)
        target = torch.argmax(target_onehot, dim=1)
        data.requires_grad = True

        output = model(data)
        init_pred = output.max(1)[1]

        # Only keep correctly predicted samples
        mask = (init_pred == target)
        if not mask.any():
            continue

        # Filter to only correct predictions
        data_to_attack = data[mask]
        target_to_attack = target[mask]

        data_to_attack.retain_grad()  # Required since it's a non-leaf tensor
        output_correct = model(data_to_attack)
        loss = F.cross_entropy(output_correct, target_to_attack)
        model.zero_grad()
        loss.backward()

        data_grad = data_to_attack.grad.data
        perturbed_data = fgsm_attack(data_to_attack, epsilon, data_grad)

        # Re-classify
        output_adv = model(perturbed_data)
        final_pred = output_adv.max(1)[1]

        # Count how many predictions are still correct
        correct += (final_pred == target_to_attack).sum().item()
        total_attacked += len(target_to_attack)

        # Save some adv examples for visualization
        if epsilon == 0 and len(adv_examples) < 5:
            adv_examples.extend(perturbed_data[:5].detach().cpu().numpy())

        final_acc = correct / total_attacked

    print(f"Epsilon: {epsilon} | Test Accuracy = {correct} / {total_attacked} = {final_acc:.4f}")
    return final_acc, adv_examples


def main():
    parser = argparse.ArgumentParser(description="FGSM attack on Wifi HAR model")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to trained model .pth file")
    parser.add_argument("--features", default=DEFAULT_FEATURES_PATH, help="Numpy features file")
    parser.add_argument("--labels", default=DEFAULT_LABELS_PATH, help="Numpy labels file")
    parser.add_argument("--epsilons", nargs="*", type=float, default=DEFAULT_EPSILONS, help="List of epsilon values")
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
    for eps in args.epsilons:
        acc, _ = test(model, device, loader, eps)
        accuracies.append(acc)

    plt.figure(figsize=(5, 5))
    plt.plot(args.epsilons, accuracies, "*-")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epsilon")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()