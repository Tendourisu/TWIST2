#!/usr/bin/env python3
"""
Minimal script to display mocap data structure from pkl files
"""
import os
import sys

import joblib
import pickle
import numpy as np


JOBLIB_LABEL = "joblib"


def load_mocap_file(file_path):
    """Load mocap data using Joblib."""
    with open(file_path, 'rb') as f:
        return joblib.load(f)


def show_mocap_structure(file_path):
    """Display the structure of a mocap pkl file."""
    print(f"\n{'='*60}")
    print(f"File: {os.path.basename(file_path)}")
    print(f"{'='*60}")

    try:
        data = load_mocap_file(file_path)
        print(f"Loader backend: {JOBLIB_LABEL}")
        print(f"Data type: {type(data)}")

        if isinstance(data, dict):
            print(f"Keys: {list(data.keys())}")
            for key, value in data.items():
                print(f"\n[{key}]:")
                if isinstance(value, np.ndarray):
                    print(f"  Shape: {value.shape}")
                    print(f"  Dtype: {value.dtype}")
                    print(f"  Min: {value.min():.4f}, Max: {value.max():.4f}")
                elif isinstance(value, dict):
                    print(f"  Dict with keys: {list(value.keys())}")
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, np.ndarray):
                            print(f"    {sub_key}: shape {sub_value.shape}, dtype {sub_value.dtype}")
                            if sub_key == 'fps':
                                with open("output.txt", "a") as file:
                                    # Write ndarray as string for better readability
                                    with np.printoptions(threshold=sub_value.size):  # ensure full array print
                                        file.write(f"{key}\n")
                                        file.write(np.array2string(sub_value) + "\n")
                        elif isinstance(sub_value, int):
                            print(f"    {sub_key}: {sub_value} (int)")
                        elif isinstance(sub_value, float):
                            print(f"    {sub_key}: {sub_value:.4f} (float)")
                else:
                    print(f"  Type: {type(value)}: {value}")

    except Exception as e:  # noqa: BLE001
        print(f"Error reading {file_path}: {e}")


def main():
    mocap_dir = "data/motions/g1_data_current"

    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if not file_path.startswith('/'):
            file_path = os.path.join(mocap_dir, file_path)
        show_mocap_structure(file_path)
    else:
        pkl_files = [f for f in os.listdir(mocap_dir) if f.endswith('.pkl')]
        print(f"Found {len(pkl_files)} pkl files in {mocap_dir}")

        if pkl_files:
            print(f"\nShowing detailed structure of: {pkl_files[0]}")
            show_mocap_structure(os.path.join(mocap_dir, pkl_files[0]))

            print(f"\n{'='*60}")
            print("SUMMARY OF ALL FILES:")
            print(f"{'='*60}")
            for pkl_file in pkl_files:
                try:
                    data = load_mocap_file(os.path.join(mocap_dir, pkl_file))
                    if isinstance(data, dict):
                        keys = list(data.keys())
                        keys_str = ', '.join(keys[:3])
                        if len(keys) > 3:
                            keys_str += '...'
                        print(f"{pkl_file:30} | Loader: {JOBLIB_LABEL:<6} | Keys: {keys_str}")
                    else:
                        print(f"{pkl_file:30} | Loader: {JOBLIB_LABEL:<6} | Type: {type(data)}")
                except Exception as e:  # noqa: BLE001
                    print(f"{pkl_file:30} | Error: {e}")


if __name__ == "__main__":
    main()
