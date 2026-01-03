import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
import pickle
from tqdm import tqdm

from transloc4d.datasets.base_datasets import TrainingTuple


def load_dataframe(base_path, dataset_name, subset, data_type):
    """
    Load the DataFrame from a specific CSV file in the dataset structure.

    Parameters:
    - base_path: The root path of the dataset.
    - dataset_name: The name of the dataset (e.g., 'ntu').
    - subset: 'train' or 'val' indicating the subset of the dataset.
    - data_type: 'ntu_org_database' or 'ntu_org_query' indicating the type of data.

    Returns:
    - A pandas DataFrame with columns ['file', 'northing', 'easting'].
    """
    csv_filename = "gps.csv"
    folder_name = "pointclouds"
    image_name = "images"

    point_folder = os.path.join(base_path, dataset_name, subset, data_type, folder_name)
    image_folder = os.path.join(base_path, dataset_name, subset, data_type, image_name)
    csv_path = os.path.join(base_path, dataset_name, subset, data_type, csv_filename)
    ext = sorted(os.listdir(point_folder))[0].split(".")[-1]

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df["file"] = df["timestamp"].apply(lambda x: os.path.join(point_folder, f"{x}.{ext}"))
    df["image_file"] = df["timestamp"].apply(lambda x: os.path.join(image_folder, f"{x}.png"))
    return df


def construct_query_dict(df_query, df_db, base_path, filename, ind_pos_r=10, ind_nonneg_r=50):
    df_combined = pd.concat([df_query, df_db]).reset_index(drop=True)

    tree_combined = KDTree(df_combined[['northing', 'easting']])
    ind_positive = tree_combined.query_radius(df_combined[['northing', 'easting']], r=ind_pos_r)
    ind_nonneg = tree_combined.query_radius(df_combined[['northing', 'easting']], r=ind_nonneg_r)
    queries = {}
    for anchor_ndx in tqdm(range(len(ind_positive)), desc="Processing"):
        positives = ind_positive[anchor_ndx]
        non_negatives = ind_nonneg[anchor_ndx]
        if len(positives) == 0 or len(non_negatives) == 0:
            continue

        anchor_pos = np.array(df_combined.iloc[anchor_ndx][['northing', 'easting']])
        timestamp = df_combined.iloc[anchor_ndx]["timestamp"]
        scan_filename = df_combined.iloc[anchor_ndx]["file"]
        assert os.path.isfile(scan_filename), 'point cloud file {} is found'.format(scan_filename)
        scan_image_filename = df_combined.iloc[anchor_ndx]["image_file"]

        # Sort ascending order
        positives = np.sort(positives)
        non_negatives = np.sort(non_negatives)

        queries[anchor_ndx] = TrainingTuple(
            id=anchor_ndx,
            timestamp=timestamp,
            rel_scan_filepath=scan_filename,
            scan_image_filename=scan_image_filename,
            positives=positives,
            non_negatives=non_negatives,
            position=anchor_pos
        )

    with open(os.path.join(base_path, filename), 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Completed: {filename}")


if __name__ == '__main__':

    #ntu train
    base_path = "/your path/RCPR-Dataset"
    # snali-radar train
    # dataset_name = "snail-n"
    # seu-radar train
    dataset_name = "seu-p"

    assert os.path.exists(base_path), f"Cannot access dataset root folder: {base_path}"
    

    df_train_query = load_dataframe(base_path, dataset_name, "train", f"query")
    df_train_db = load_dataframe(base_path, dataset_name, "train", f"database")

    construct_query_dict(df_train_query, df_train_db, base_path, f"train_queries_rc_{dataset_name}.pickle", ind_pos_r=10, ind_nonneg_r=50)