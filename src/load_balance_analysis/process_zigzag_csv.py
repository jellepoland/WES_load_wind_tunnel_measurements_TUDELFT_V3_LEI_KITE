import pandas as pd
import numpy as np
from pathlib import Path
import os
from load_balance_analysis.functions_utils import project_dir
from load_balance_analysis.functions_processing import processing_raw_lvm_data_into_csv


def merging_zigzag_csv_files(
    zigzag_data_dir: Path, save_path_merged_zigzag: Path
) -> None:

    all_dfs = []
    for file in os.listdir(zigzag_data_dir):
        file_path = Path(zigzag_data_dir) / file
        print(f"\nfile_path: {file_path}")
        if not file_path.is_file():
            continue  # Skip directories
        if "raw" in file or "lvm" in file:
            continue
        print(f"\n--> Processing file: {file}")
        df = pd.read_csv(file_path)
        all_dfs.append(df)

    print(f"all_dfs: {all_dfs}")

    # Concatenate all the dataframes
    merged_df = pd.concat(all_dfs, ignore_index=True)

    # Save the merged dataframe to a csv file
    merged_df.to_csv(save_path_merged_zigzag, index=False)


def main():
    # processing all the folders for the normal
    support_struc_aero_interp_coeffs_path = (
        Path(project_dir) / "processed_data" / "interpolation_coefficients.csv"
    )
    is_kite = True
    is_zigzag = True
    print(f"\n Processing all the folders")
    for folder in os.listdir(Path(project_dir) / "processed_data" / "zigzag_csv"):
        print(f"\nfolder: {folder}")
        if "alpha" not in folder:
            continue
        folder_dir = Path(project_dir) / "processed_data" / "zigzag_csv" / folder
        print(f"\nfolder_dir: {folder_dir}")
        processing_raw_lvm_data_into_csv(
            folder_dir,
            is_kite,
            is_zigzag,
            support_struc_aero_interp_coeffs_path,
        )

    # Merging raw files
    processed_data_zigzag_dir = (
        Path(project_dir) / "processed_data" / "zigzag_csv" / "alpha_9.9"
    )
    save_path_lvm_data_processed = (
        Path(processed_data_zigzag_dir) / "lvm_data_processed.csv"
    )
    merging_zigzag_csv_files(processed_data_zigzag_dir, save_path_lvm_data_processed)


if __name__ == "__main__":
    main()
