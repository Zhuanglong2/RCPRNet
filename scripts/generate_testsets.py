import os
import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
import pickle
from tqdm import tqdm


def load_dataframe(base_path, dataset_name, subset, data_type):
    """
    Load the DataFrame from a specific CSV file in the dataset structure.
    """
    csv_filename = "gps.csv"
    folder_name = "pointclouds"
    image_name = "images"
    csv_path = os.path.join(base_path, dataset_name, subset, data_type, csv_filename)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    point_folder = os.path.join(base_path, dataset_name, subset, data_type, folder_name)
    image_folder = os.path.join(base_path, dataset_name, subset, data_type, image_name)
    ext = sorted(os.listdir(point_folder))[0].split(".")[-1]
    df = pd.read_csv(csv_path)
    df["file"] = df["timestamp"].apply(lambda x: os.path.join(point_folder, f"{x}.{ext}"))
    # Adjust column names to match 'northing', 'easting' if necessary
    df["image_file"] = df["timestamp"].apply(lambda x: os.path.join(image_folder, f"{x}.png"))
    return df


def output_to_file(output, base_path, filename):
    """
    Saves the given data to a pickle file, similar to the first script.
    """
    file_path = os.path.join(base_path, filename)
    with open(file_path, "wb") as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Completed: {filename}")


def construct_query_and_database_sets(base_path, datasets_name, subsets, valDist=25):
    """
    Constructs and saves the query and database sets for each subset in the dataset,
    following the logic and output format of the first script.
    """
    for dataset_name in datasets_name:
        for subset in subsets:
            df_query = load_dataframe(base_path, dataset_name, subset, f"query")
            df_database = load_dataframe(base_path, dataset_name, subset, f"database")

            # Build KDTree for the database
            tree_database = KDTree(df_database[["northing", "easting"]])

            # Initialize containers for the structured output
            database_sets = []
            test_sets = []

            # Process queries to find positive matches within the database
            for index, row in tqdm(df_query.iterrows(), total=df_query.shape[0], desc=f"Processing {subset} subset",):
                coor = np.array([[row["northing"], row["easting"]]])
                # Radius search for positives
                indices = tree_database.query_radius(coor, r=valDist)[0].tolist()

                # Assuming the same structuring as your first script, adjust as necessary
                test = {
                    "file": row["file"],
                    "image_file": row["image_file"],
                    "northing": row["northing"],
                    "easting": row["easting"],
                    "positives": indices,  # Indices of the positive matches
                }
                test_sets.append(test)

            # Process the database entries similarly if needed
            for _, row in df_database.iterrows():
                database = {
                    "file": row["file"],
                    "image_file": row["image_file"],
                    "northing": row["northing"],
                    "easting": row["easting"],
                }
                database_sets.append(database)

            # Output to files, following naming convention similar to the first script
            output_to_file(
                database_sets,
                base_path,
                f"{dataset_name}_{subset}_evaluation_database_rc_{valDist}.pickle",
            )
            output_to_file(
                test_sets,
                base_path,
                f"{dataset_name}_{subset}_evaluation_query_rc_{valDist}.pickle",
            )

if __name__ == "__main__":
    tasks = [
        #Snail-radar
        {
            "base_path": "/your path/RCPR-Dataset",
            "datasets_name": ["snail-d"],
            "subsets": ["val"],
            "valDist": 25,
        },
        {
            "base_path": "/your path/RCPR-Dataset",
            "datasets_name": ["snail-lrn","snail-tv"],
            "subsets": ["test"],
            "valDist": 25,
        },
        #SEU
        {
            "base_path": "/your path/RCPR-Dataset",
            "datasets_name": ["seu-af"],
            "subsets": ["val"],
            "valDist": 25,
        },
        {
            "base_path": "/your path/RCPR-Dataset",
            "datasets_name": ["seu-af", "seu-p"],
            "subsets": ["test"],
            "valDist": 25,
        }
    ]

    for task in tasks:
        base_path = task["base_path"]
        datasets_name = task["datasets_name"]
        subsets = task["subsets"]
        valDist = task.get("valDist", 25)

        # Ensure the dataset's base path exists
        assert os.path.exists(base_path), f"Cannot access dataset root folder: {base_path}"

        # Construct and save the query and database sets
        construct_query_and_database_sets(base_path, datasets_name, subsets, valDist=valDist)
