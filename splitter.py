# This script organizes the dataset by moving label files to their respective train/val folders.

import os
import shutil
def splitter():
    """
    This function organizes the dataset by moving label files to their respective train/val folders.
    It assumes that the label files are named according to the images they correspond to.
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # "final project" directory
    dataset_path = os.path.join(BASE_DIR, "dataset")
    images_train = os.path.join(dataset_path, "images/train")
    images_val = os.path.join(dataset_path, "images/val")
    labels_path = os.path.join(dataset_path, "labels")
    labels_train = os.path.join(labels_path, "train")
    labels_val = os.path.join(labels_path, "val")

    # Ensure train/val label folders exist
    os.makedirs(labels_train, exist_ok=True)
    os.makedirs(labels_val, exist_ok=True)

    # Move corresponding label files to train/val folders
    for img_folder, lbl_folder in [(images_train, labels_train), (images_val, labels_val)]:
        for img_file in os.listdir(img_folder):
            label_file = os.path.splitext(img_file)[0] + ".txt"
            src_label = os.path.join(labels_path, label_file)
            dest_label = os.path.join(lbl_folder, label_file)
            
            if os.path.exists(src_label):
                shutil.move(src_label, dest_label)
            else:
                print(f"Warning: Label for {img_file} not found.")

    print("✅ Labels successfully organized into train/val folders.")