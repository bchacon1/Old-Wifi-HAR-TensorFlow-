import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pytorch_model import WifiHARLSTM, n_input, n_hidden, n_classes, window_size


def load_data(features_path: str, labels_path: str):
    features = np.load(features_path)
    labels = np.load(labels_path)
    dataset = TensorDataset(
        torch.from_numpy(features).float(),
        torch.from_numpy(labels).float(),
    )
    return DataLoader(dataset, batch_size=1, shuffle=True)


def fgsm_attack(data, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed = data + epsilon * sign_data_grad
    return perturbed


def test(model, device, loader, epsilon):
    correct = 0
    for data, target_onehot in loader:
        data, target_onehot = data.to(device), target_onehot.to(device)
        target = torch.argmax(target_onehot, dim=1)
        data.requires_grad = True
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
        if init_pred.item() != target.item():
            continue
        loss = F.cross_entropy(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            correct += 1
    final_acc = correct / float(len(loader))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(loader)} = {final_acc:.4f}")
    return final_acc


def main():
    parser = argparse.ArgumentParser(description="FGSM attack on Wifi HAR model")
    parser.add_argument("--model", required=True, help="Path to trained model .pth file")
    parser.add_argument(
        "--features", default="input_files/all_features_full.npy", help="Numpy features file"
    )
    parser.add_argument(
        "--labels", default="input_files/all_labels_full.npy", help="Numpy labels file"
    )
    parser.add_argument(
        "--epsilons",
        nargs="*",
        type=float,
        default=[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
        help="List of epsilon values",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = load_data(args.features, args.labels)
    model = WifiHARLSTM(n_input, n_hidden, n_classes).to(device)
    state = torch.load(args.model, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()

    for eps in args.epsilons:
        test(model, device, loader, eps)


if __name__ == "__main__":
    main()

