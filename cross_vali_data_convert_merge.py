"""Utilities for converting raw CSI CSV files to windowed feature and label datasets.

This script mirrors the preprocessing used in the original TensorFlow code. It
reads raw CSI input files and their corresponding annotation files, then
generates sliding windows of features and labels. Each window contains
``raw_window_size`` samples with a stride of ``slide_size``. The resulting arrays are
saved as CSVs for later training. In addition, the combined dataset is stored as
``.npy`` files so PyTorch training scripts can load the data quickly.
"""

import numpy as np
import csv
import glob
import os

raw_window_size = 1000
window_size = raw_window_size // 2  # 1000 Hz downsampled to 500 Hz
threshold = 60
slide_size = 200  # less than raw_window_size


def dataimport(path1: str, path2: str):
    """Import input/annotation csv files and produce feature and label arrays."""
    xx = np.empty((0, raw_window_size, 90), float)
    yy = np.empty((0, 8), float)

    # --- Input CSI data ---
    input_csv_files = sorted(glob.glob(path1))
    for f in input_csv_files:
        print("input_file_name=", f)
        data = [[float(elm) for elm in v] for v in csv.reader(open(f, "r"))]
        tmp1 = np.array(data)
        x2 = np.empty((0, raw_window_size, 90), float)

        k = 0
        while k <= (len(tmp1) + 1 - 2 * raw_window_size):
            x = np.dstack(np.array(tmp1[k:k + raw_window_size, 1:91]).T)
            x2 = np.concatenate((x2, x), axis=0)
            k += slide_size

        xx = np.concatenate((xx, x2), axis=0)

    # --- Annotation data ---
    annotation_csv_files = sorted(glob.glob(path2))
    for ff in annotation_csv_files:
        print("annotation_file_name=", ff)
        ano_data = [[str(elm) for elm in v] for v in csv.reader(open(ff, "r"))]
        tmp2 = np.array(ano_data)

        y = np.zeros(((len(tmp2) + 1 - 2 * raw_window_size) // slide_size + 1, 8))
        k = 0
        while k <= (len(tmp2) + 1 - 2 * raw_window_size):
            y_pre = np.stack(np.array(tmp2[k:k + raw_window_size]))
            counts = {
                "bed": 0,
                "fall": 0,
                "walk": 0,
                "pickup": 0,
                "run": 0,
                "sitdown": 0,
                "standup": 0,
                "no": 0,
            }
            for j in range(raw_window_size):
                lbl = y_pre[j]
                if lbl in counts:
                    counts[lbl] += 1
                else:
                    counts["no"] += 1
            if counts["bed"] > raw_window_size * threshold / 100:
                y[k // slide_size, :] = np.array([0, 1, 0, 0, 0, 0, 0, 0])
            elif counts["fall"] > raw_window_size * threshold / 100:
                y[k // slide_size, :] = np.array([0, 0, 1, 0, 0, 0, 0, 0])
            elif counts["walk"] > raw_window_size * threshold / 100:
                y[k // slide_size, :] = np.array([0, 0, 0, 1, 0, 0, 0, 0])
            elif counts["pickup"] > raw_window_size * threshold / 100:
                y[k // slide_size, :] = np.array([0, 0, 0, 0, 1, 0, 0, 0])
            elif counts["run"] > raw_window_size * threshold / 100:
                y[k // slide_size, :] = np.array([0, 0, 0, 0, 0, 1, 0, 0])
            elif counts["sitdown"] > raw_window_size * threshold / 100:
                y[k // slide_size, :] = np.array([0, 0, 0, 0, 0, 0, 1, 0])
            elif counts["standup"] > raw_window_size * threshold / 100:
                y[k // slide_size, :] = np.array([0, 0, 0, 0, 0, 0, 0, 1])
            else:
                y[k // slide_size, :] = np.array([2, 0, 0, 0, 0, 0, 0, 0])
            k += slide_size

        yy = np.concatenate((yy, y), axis=0)

    # Remove "NoActivity" rows and drop the first column
    mask = yy[:, 0] != 2
    xx = xx[mask]
    yy = yy[mask, 1:]
    # Downsample from 1000 Hz to 500 Hz to match training input
    xx = xx[:, ::2, :]
    print(xx.shape, yy.shape)
    return xx, yy


if __name__ == "__main__":
    os.makedirs("input_files", exist_ok=True)

    labels = ["bed", "fall", "pickup", "run", "sitdown", "standup", "walk"]
    all_features = []
    all_labels = []

    for label in labels:
        filepath1 = f"./Dataset/Data/input_*{label}*.csv"
        filepath2 = f"./Dataset/Data/annotation_*{label}*.csv"
        output_x = f"./input_files/xx_{raw_window_size}_{threshold}_{label}.csv"
        output_y = f"./input_files/yy_{raw_window_size}_{threshold}_{label}.csv"

        x, y = dataimport(filepath1, filepath2)

        # Save individual CSV files for compatibility
        with open(output_x, "w") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerows(x.reshape(len(x), -1))
        with open(output_y, "w") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerows(y)

        all_features.append(x)
        all_labels.append(y)
        print(label + " finish!")

    all_features_full = np.concatenate(all_features, axis=0).astype(np.float32)
    all_labels_full = np.concatenate(all_labels, axis=0).astype(np.float32)
    np.save(os.path.join("input_files", "all_features_full.npy"), all_features_full)
    np.save(os.path.join("input_files", "all_labels_full.npy"), all_labels_full)
    print("Saved combined dataset to input_files/all_features_full.npy and all_labels_full.npy")
