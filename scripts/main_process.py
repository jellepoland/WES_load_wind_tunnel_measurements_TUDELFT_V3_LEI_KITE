from pathlib import Path
from load_balance_analysis import process_bundling_beta_0
from load_balance_analysis import process_support_struc_aero_interp_coeffs
from load_balance_analysis import process_normal_csv
from load_balance_analysis import process_zigzag_csv
from load_balance_analysis import process_raw_lvm_with_labbook_into_df
from load_balance_analysis import plot_zigzag
from load_balance_analysis import process_vsm
from load_balance_analysis import process_uncertainty_table
from load_balance_analysis.functions_utils import project_dir


def main():
    # Creating necessary folders
    for path in [
        Path(project_dir) / "processed_data" / "polar_data",
        Path(project_dir) / "processed_data" / "normal_csv",
        Path(project_dir) / "processed_data" / "uncertainty_table",
        Path(project_dir) / "processed_data" / "without_csv",
        Path(project_dir) / "processed_data" / "zigzag_csv",
        Path(project_dir) / "results" / "tables",
    ]:
        path.mkdir(parents=True, exist_ok=True)

    process_raw_lvm_with_labbook_into_df.main()
    process_support_struc_aero_interp_coeffs.main()
    process_normal_csv.main()
    process_zigzag_csv.main()

    # takes ages..
    # process_vsm.main()
    process_bundling_beta_0.main()

    ## this function takes long
    process_uncertainty_table.main(project_dir)


if __name__ == "__main__":
    main()
