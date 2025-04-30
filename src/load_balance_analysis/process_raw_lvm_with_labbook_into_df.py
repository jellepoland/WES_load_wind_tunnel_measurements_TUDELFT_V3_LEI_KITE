import os
from pathlib import Path
import pandas as pd
import numpy as np
from load_balance_analysis.functions_utils import project_dir


def read_lvm(filename: str) -> pd.DataFrame:
    # Read the entire data file
    df = pd.read_csv(filename, skiprows=21, delimiter="\t", engine="python")

    # Print debugging information
    print(f"File: {filename}")
    print(f"Number of rows: {len(df)}")

    # Drop the last column if it's empty or unnecessary
    if df.columns[-1].strip() == "":
        df.drop(columns=df.columns[-1], inplace=True)

    # Rename columns, assuming we have the correct number after dropping empty column
    expected_columns = ["time", "F_X", "F_Y", "F_Z", "M_X", "M_Y", "M_Z"]
    if len(df.columns) == len(expected_columns):
        df.columns = expected_columns
    else:
        print(
            f"Warning: Expected {len(expected_columns)} columns, but got {len(df.columns)}"
        )
        df = df.iloc[:, :7]
        df.columns = expected_columns

    # Extract the filename
    base_filename = os.path.basename(filename)
    df["Filename"] = base_filename.replace(".lvm", "")

    # Calculate sample index more flexibly
    total_rows = len(df)
    if total_rows % 19800 != 0:
        print(f"Warning: Number of rows ({total_rows}) is not divisible by 19800")
        actual_sample_size = total_rows  # treat the entire file as one sample
        num_samples = 1
    else:
        actual_sample_size = 19800
        num_samples = total_rows // actual_sample_size

    # Create sample index
    sample_indices = []
    for i in range(num_samples):
        start_idx = i * actual_sample_size
        end_idx = start_idx + actual_sample_size
        sample_indices.extend([i] * (end_idx - start_idx))

    # Ensure sample_indices matches the DataFrame length
    if len(sample_indices) != len(df):
        print(
            f"Warning: sample_indices length ({len(sample_indices)}) doesn't match DataFrame length ({len(df)})"
        )
        # Adjust sample_indices to match DataFrame length
        if len(sample_indices) < len(df):
            sample_indices.extend([num_samples - 1] * (len(df) - len(sample_indices)))
        else:
            sample_indices = sample_indices[: len(df)]

    df["sample_index"] = sample_indices

    print(f"Created {num_samples} samples")
    return df


def read_all_lvm_into_df(labbook_df: pd.DataFrame, data_dir: Path) -> list:

    # Remove rows where wind speed is zero from labbook
    # labbook_df = labbook_df[labbook_df["vw"] > 1.1]

    # print(f"Labbook: {labbook_df}")

    # zigzag_data_dir = Path(project_dir) / "data" / "zigzag"
    all_data = []

    # Extract only the lvm files
    lvm_files = [f for f in os.listdir(data_dir) if f.endswith(".lvm")]

    # Loop through each file in the zigzag directory
    for file in lvm_files:
        print(f"\nProcessing file: {file}")

        # Read the lvm file
        lvm_file_path = Path(data_dir) / file
        df = read_lvm(lvm_file_path)

        # Strip lvm from file
        filename = lvm_file_path.stem
        # Strip _unsteady from filename
        filename = filename.replace("_unsteady", "")
        print(f"Filename: {filename}")

        # Filter labbook for where folder matches the filename
        matching_rows = labbook_df[labbook_df["Filename"] == filename]

        # print(f"Matching rows: {matching_rows}")
        n_rows = len(matching_rows)
        # print(f"Number of matching rows: {n_rows}")

        # Calculate sample index based on the number of rows and the assumption that each sample has 19800 entries
        # print(f"len(df): {len(df)}")
        num_samples = len(df) // 19800
        # print(f"Number of samples: {num_samples}")

        if n_rows == 1:
            if num_samples == 1:
                df_sample = df.copy()
            elif num_samples == 2:
                df_sample = df.iloc[19800:].copy()

            last_matching_row = matching_rows.iloc[-1]  # Get the last occurrence
            properties_to_add = last_matching_row.to_dict()
            # print(f"properties_to_add: {properties_to_add}")

            # Add properties from labbook to each row in df
            for col, value in properties_to_add.items():
                df_sample[col] = (
                    value  # This will create new columns in df with properties from labbook_df
                )

            all_data.append(df_sample)
        else:
            # print(f"\nMore than 1 matching row")
            # Create a new DataFrame for each sample
            for i in range(num_samples):
                # Selecting the sample from the DataFrame
                df_sample = df.iloc[i * 19800 : (i + 1) * 19800].copy()

                # Finding the matching row
                row = matching_rows.iloc[i]
                # print(f"Row: {row}")
                properties_to_add = row.to_dict()
                # print(f"properties_to_add: {properties_to_add}")

                # Add properties from labbook to each row in df
                for col, value in properties_to_add.items():
                    df_sample[col] = (
                        value  # This will create new columns in df with properties from labbook_df
                    )

                all_data.append(df_sample)
    df_all = pd.concat(all_data)

    # Dropping the 'time' column and 'Unnamed' columns after Unnamed: 11
    drop_column_list = [
        "time",
        # "F_X",
        # "F_Y",
        # "F_Z",
        # "M_X",
        # "M_Y",
        # "M_Z",
        # "Filename",
        "sample_index",
        # "Date",
        "Folder",
        # "config",
        # "aoa",
        # "sideslip",
        # "vw",
        # "Dpa",
        # "Pressure",
        # "Temp",
        # "Density",
        "Unnamed: 11",
        "rows",
        "Unnamed: 13",
        "Unnamed: 14",
        "Unnamed: 15",
        "Unnamed: 16",
        "Unnamed: 17",
        "Unnamed: 18",
        "Unnamed: 19",
        "Unnamed: 20",
        "Unnamed: 21",
    ]
    for col in drop_column_list:
        if col in df_all.columns:
            df_all.drop(columns=[col], inplace=True)

    # Define the desired column order
    desired_order = [
        "Filename",
        "vw",
        "aoa",
        "sideslip",
        "Date",
        "F_X",
        "F_Y",
        "F_Z",
        "M_X",
        "M_Y",
        "M_Z",
        "Dpa",
        "Pressure",
        "Temp",
        "Density",
    ]

    # Reorder the DataFrame columns
    df_reordered = df_all[desired_order]

    return df_reordered


def read_labbook_into_df(labbook_path: Path) -> pd.DataFrame:
    return pd.read_csv(labbook_path, delimiter=";")


def reading_all_folders(
    labbook_path: Path,
    parent_folder_dir: Path,
    save_parent_dir: Path,
    delta_aoa_rod_to_alpha: float,
):

    # loading labbook
    labbook_df = read_labbook_into_df(labbook_path)

    # Ensuring the existence of parent_folder
    if not os.path.exists(parent_folder_dir):
        os.mkdir(parent_folder_dir)

    for folder_name in os.listdir(parent_folder_dir):
        print(f"\n READING folder_name: {folder_name}")
        folder_dir = Path(parent_folder_dir) / folder_name
        df_folder = read_all_lvm_into_df(labbook_df, folder_dir)

        ## Making a directory for this folder
        # Stripping the angle
        if isinstance(df_folder["aoa"].unique()[0], np.float64):
            aoa_value = float(df_folder["aoa"].unique()[0])
        else:
            aoa_value = float(df_folder["aoa"].unique()[0].replace(",", "."))
        # df_folder["aoa"] = df_folder["aoa"].str.replace("deg", "")
        # aoa_value = folder_name.split("_")[-1]
        ## Changing alpha by the offsett deg
        aoa_value_corrected = float(aoa_value) - delta_aoa_rod_to_alpha
        folder_name_corrected = f"alpha_{aoa_value_corrected:.1f}"
        print(f"\nSAVING folder_name_corrected: {folder_name_corrected}")
        output_folder_dir = Path(save_parent_dir) / folder_name_corrected
        os.makedirs(output_folder_dir, exist_ok=True)

        # Finding all the unique runs within the folder
        unique_filenames = df_folder["Filename"].unique()
        # print(f"unique_filenames: {unique_filenames}")

        # Saving each filename as a separate CSV file
        for filename in unique_filenames:
            df_filename = df_folder[df_folder["Filename"] == filename]
            output_path = output_folder_dir / f"raw_{filename}.csv"

            # TODO: Changing all the sideslip values from sign
            # df_filename["sideslip"] = -df_filename["sideslip"]
            df_filename.to_csv(output_path, index=False)


def main():
    delta_aoa_rod_to_alpha = 6.3  # 7.25

    ## Doing this for the normal data
    labbook_path = Path(project_dir) / "data" / "labbook.csv"
    parent_folder_dir = Path(project_dir) / "data" / "normal"
    save_parent_dir = Path(project_dir) / "processed_data" / "normal_csv"
    reading_all_folders(
        labbook_path, parent_folder_dir, save_parent_dir, delta_aoa_rod_to_alpha
    )

    ## Doing this for the without data
    labbook_path = Path(project_dir) / "data" / "labbook.csv"
    parent_folder_dir = Path(project_dir) / "data" / "without"
    save_parent_dir = Path(project_dir) / "processed_data" / "without_csv"
    reading_all_folders(
        labbook_path, parent_folder_dir, save_parent_dir, delta_aoa_rod_to_alpha
    )

    ## Doing this for the zigzag data
    parent_folder_dir = Path(project_dir) / "data" / "zigzag"
    save_parent_dir = Path(project_dir) / "processed_data" / "zigzag_csv"
    labbook_zz_path = Path(project_dir) / "data" / "labbook_zz.csv"
    reading_all_folders(
        labbook_zz_path, parent_folder_dir, save_parent_dir, delta_aoa_rod_to_alpha
    )


if __name__ == "__main__":
    main()
