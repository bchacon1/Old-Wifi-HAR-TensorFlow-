"""Utilities for converting raw CSI CSV files to windowed feature/label
datasets.

This script mirrors the preprocessing used in the original TensorFlow code. It
reads raw CSI input files and their corresponding annotation files, then
generates sliding windows of features and labels.  Each window contains
``window_size`` samples with a stride of ``slide_size``.  The resulting arrays
are saved as CSVs for later training.  In addition to the per-activity CSV
files, the script now also stores the combined dataset as ``.npy`` files so
subsequent training scripts can load the data directly without re-parsing the
CSVs.
"""

import numpy as np
import csv
import glob
import os

def _count_windows(num_rows):
    """Return number of sliding windows for a sequence of length ``num_rows``."""
    return (num_rows + 1 - 2 * window_size) // slide_size + 1

window_size = 1000
threshold = 60
slide_size = 200 #less than window_size!!!

def dataimport(path1, path2):

    xx = np.empty([0,window_size,90],float)
    yy = np.empty([0,8],float)

    ###Input data###
    #data import from csv
    input_csv_files = sorted(glob.glob(path1))
    for f in input_csv_files:
        print("input_file_name=",f)
        data = [[ float(elm) for elm in v] for v in csv.reader(open(f, "r"))]
        tmp1 = np.array(data)
        x2 =np.empty([0,window_size,90],float)

        #data import by slide window
        k = 0
        while k <= (len(tmp1) + 1 - 2 * window_size):
            x = np.dstack(np.array(tmp1[k:k+window_size, 1:91]).T)
            x2 = np.concatenate((x2, x), axis=0)
            k += slide_size

        xx = np.concatenate((xx, x2), axis=0)
        # ``xx`` has shape [num_samples, window_size, 90] here.  Keep this
        # 3-D representation so it can be directly saved to ``.npy`` and later
        # downsampled by the training script if necessary.

    ###Annotation data###
    #data import from csv
    annotation_csv_files = sorted(glob.glob(path2))
    for ff in annotation_csv_files:
        print("annotation_file_name=",ff)
        ano_data = [[ str(elm) for elm in v] for v in csv.reader(open(ff,"r"))]
        tmp2 = np.array(ano_data)

        #data import by slide window
        y = np.zeros(((len(tmp2) + 1 - 2 * window_size)//slide_size+1,8))
        k = 0
        while k <= (len(tmp2) + 1 - 2 * window_size):
            y_pre = np.stack(np.array(tmp2[k:k+window_size]))
            bed = 0
            fall = 0
            walk = 0
            pickup = 0
            run = 0
            sitdown = 0
            standup = 0
            noactivity = 0
            for j in range(window_size):
                if y_pre[j] == "bed":
                    bed += 1
                elif y_pre[j] == "fall":
                    fall += 1
                elif y_pre[j] == "walk":
                    walk += 1
                elif y_pre[j] == "pickup":
                    pickup += 1
                elif y_pre[j] == "run":
                    run += 1
                elif y_pre[j] == "sitdown":
                    sitdown += 1
                elif y_pre[j] == "standup":
                    standup += 1
                else:
                    noactivity += 1
#change by Brian here, had to change k / slide_size to // because / generates float points and numpy requires integer indicies
            if bed > window_size * threshold / 100:
                y[k // slide_size,:] = np.array([0,1,0,0,0,0,0,0])
            elif fall > window_size * threshold / 100:
                y[k // slide_size,:] = np.array([0,0,1,0,0,0,0,0])
            elif walk > window_size * threshold / 100:
                y[k // slide_size,:] = np.array([0,0,0,1,0,0,0,0])
            elif pickup > window_size * threshold / 100:
                y[k // slide_size,:] = np.array([0,0,0,0,1,0,0,0])
            elif run > window_size * threshold / 100:
                y[k // slide_size,:] = np.array([0,0,0,0,0,1,0,0])
            elif sitdown > window_size * threshold / 100:
                y[k // slide_size,:] = np.array([0,0,0,0,0,0,1,0])
            elif standup > window_size * threshold / 100:
                y[k // slide_size,:] = np.array([0,0,0,0,0,0,0,1])
            else:
                y[k // slide_size,:] = np.array([2,0,0,0,0,0,0,0])
            k += slide_size

        yy = np.concatenate((yy, y),axis=0)
    print(xx.shape,yy.shape)
    return (xx, yy)


#### Main ####
if not os.path.exists("input_files/"):
    os.makedirs("input_files/")

labels = ["bed", "fall", "pickup", "run", "sitdown", "standup", "walk"]

# --- Determine total number of windows across all activities ---
label_window_counts = []
for lbl in labels:
    path_pattern = f"./Dataset/Data/annotation_*{lbl}*.csv"
    count = 0
    for anno_file in sorted(glob.glob(path_pattern)):
        num_lines = sum(1 for _ in open(anno_file))
        count += _count_windows(num_lines)
    label_window_counts.append(count)

total_windows = sum(label_window_counts)

# Preallocate memmap arrays so we never keep all data in memory
features_mm = np.lib.format.open_memmap(
    os.path.join("input_files", "all_features_full.npy"),
    mode="w+",
    dtype=np.float32,
    shape=(total_windows, window_size, 90),
)
labels_mm = np.lib.format.open_memmap(
    os.path.join("input_files", "all_labels_full.npy"),
    mode="w+",
    dtype=np.float32,
    shape=(total_windows, 8),
)

start_idx = 0
for lbl, lbl_count in zip(labels, label_window_counts):
    filepath1 = f"./Dataset/Data/input_*{lbl}*.csv"
    filepath2 = f"./Dataset/Data/annotation_*{lbl}*.csv"
    out_x_csv = f"./input_files/xx_{window_size}_{threshold}_{lbl}.csv"
    out_y_csv = f"./input_files/yy_{window_size}_{threshold}_{lbl}.csv"

    x, y = dataimport(filepath1, filepath2)

    # Save individual CSV files for backward compatibility
    with open(out_x_csv, "w") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(x.reshape(len(x), -1))
    with open(out_y_csv, "w") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(y)

    # Store in preallocated memmap
    end_idx = start_idx + len(x)
    features_mm[start_idx:end_idx] = x.astype(np.float32)
    labels_mm[start_idx:end_idx] = y.astype(np.float32)
    start_idx = end_idx

    print(lbl + "finish!")

# Flush memmaps to disk
features_mm.flush()
labels_mm.flush()
print(
    "Saved combined dataset to input_files/all_features_full.npy and all_labels_full.npy"
)
