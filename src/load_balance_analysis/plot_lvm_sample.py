import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from load_balance_analysis.functions_utils import (
    saving_pdf_and_pdf_tex,
    project_dir,
    apply_angle_wind_tunnel_corrections_to_df,
)
from load_balance_analysis.functions_statistics import (
    hac_newey_west_confidence_interval,
)
import sys

sys.path.append(str(Path(project_dir).parent / "scripts"))
from plot_styling import plot_on_ax, set_plot_style


def read_lvm(filename: str) -> pd.DataFrame:
    """Read LVM file and return DataFrame with proper columns"""
    df = pd.read_csv(filename, skiprows=21, delimiter="\t", engine="python")

    print(f"File: {filename}")
    print(f"Number of rows: {len(df)}")

    # Drop the last column if it's empty
    if df.columns[-1].strip() == "":
        df.drop(columns=df.columns[-1], inplace=True)

    # Rename columns
    expected_columns = ["time", "F_X", "F_Y", "F_Z", "M_X", "M_Y", "M_Z"]
    if len(df.columns) == len(expected_columns):
        df.columns = expected_columns
    else:
        print(
            f"Warning: Expected {len(expected_columns)} columns, but got {len(df.columns)}"
        )
        df = df.iloc[:, :7]
        df.columns = expected_columns

    return df


def read_labbook_into_df(labbook_path: Path) -> pd.DataFrame:
    """Read labbook CSV file"""
    return pd.read_csv(labbook_path, delimiter=";")


def process_lvm_with_labbook(lvm_path: Path, labbook_path: Path) -> pd.DataFrame:
    """Process LVM file with labbook information and extract sideslip=0 data"""
    # Read LVM data
    df = read_lvm(str(lvm_path))

    # Read labbook
    labbook_df = read_labbook_into_df(labbook_path)

    # Extract filename without extension and _unsteady suffix
    filename = lvm_path.stem.replace("_unsteady", "")
    print(f"Processing filename: {filename}")

    # Find all matching rows in labbook for this filename
    matching_rows = labbook_df[labbook_df["Filename"] == filename]

    if len(matching_rows) == 0:
        raise ValueError(f"No matching row found in labbook for {filename}")

    print(f"Found {len(matching_rows)} samples in labbook for {filename}")

    # Find the sample index for sideslip = 0
    sideslip_zero_rows = matching_rows[matching_rows["sideslip"] == 0]

    if len(sideslip_zero_rows) == 0:
        print("Warning: No sideslip = 0 samples found!")
        print("Available sideslip values:", sorted(matching_rows["sideslip"].unique()))
        # Use the first sample as fallback
        sample_info = matching_rows.iloc[0]
        sample_index = 1  # Default to first sample
    else:
        sample_info = sideslip_zero_rows.iloc[0]
        sample_index = int(sample_info["rows"])  # This should be the sample index

    print(
        f"Using sample index {sample_index} with sideslip = {sample_info['sideslip']}"
    )

    # Calculate which rows in the LVM correspond to this sample
    # Based on the processing code, each sample appears to be 19800 rows
    rows_per_sample = 19800
    start_row = (sample_index - 1) * rows_per_sample  # Convert to 0-based indexing
    end_row = start_row + rows_per_sample

    print(f"Extracting rows {start_row} to {end_row-1} from total {len(df)} rows")

    # Extract the specific sample data
    if end_row > len(df):
        print(f"Warning: Calculated end row {end_row} exceeds data length {len(df)}")
        df_sample = df.iloc[start_row:].copy()
    else:
        df_sample = df.iloc[start_row:end_row].copy()

    print(f"Extracted {len(df_sample)} rows for analysis")

    # Add labbook properties to DataFrame
    for col, value in sample_info.to_dict().items():
        if col not in ["rows"]:  # Skip the rows column
            df_sample[col] = value

    # Apply wind tunnel corrections
    df_sample = apply_angle_wind_tunnel_corrections_to_df(df_sample)

    # Debug: print available columns
    print(f"Available columns after processing: {list(df_sample.columns)}")
    print(f"Sample of data:")
    print(df_sample.head())
    print(
        f"Sideslip value: {df_sample['sideslip'].iloc[0] if 'sideslip' in df_sample.columns else 'N/A'}"
    )

    return df_sample


def filter_sideslip_zero(df: pd.DataFrame) -> pd.DataFrame:
    """Filter DataFrame to only include samples where sideslip = 0"""
    if "sideslip" not in df.columns:
        print("Warning: 'sideslip' column not found. Returning original DataFrame.")
        return df

    # Filter for sideslip = 0 (with some tolerance for floating point comparison)
    sideslip_zero_mask = abs(df["sideslip"]) < 0.1  # Allow small tolerance
    df_filtered = df[sideslip_zero_mask].copy()

    print(f"\nFiltering for sideslip ≈ 0:")
    print(f"Original samples: {len(df)}")
    print(f"Filtered samples (sideslip ≈ 0): {len(df_filtered)}")
    print(f"Percentage kept: {len(df_filtered)/len(df)*100:.1f}%")

    if len(df_filtered) == 0:
        print("Warning: No samples found with sideslip ≈ 0!")
        return df

    return df_filtered


def calculate_block_averages(data: pd.Series, block_size: int) -> tuple:
    """Calculate block averages for a time series, including partial last block"""
    n_complete_blocks = len(data) // block_size
    block_means = []
    block_indices = []

    # Process complete blocks
    for i in range(n_complete_blocks):
        start_idx = i * block_size
        end_idx = start_idx + block_size
        block_mean = data.iloc[start_idx:end_idx].mean()
        block_means.append(block_mean)
        block_indices.append(i + 1)

    # Process remaining samples if any
    remaining_samples = len(data) % block_size
    if remaining_samples > 0:
        start_idx = n_complete_blocks * block_size
        block_mean = data.iloc[start_idx:].mean()
        block_means.append(block_mean)
        block_indices.append(n_complete_blocks + 1)

    return np.array(block_indices), np.array(block_means)


def calculate_running_averages(data: pd.Series, window_size: int) -> tuple:
    """Calculate cumulative averages for a time series"""
    # Calculate cumulative averages: average of samples 1 to N for each N
    cumulative_means = data.expanding().mean()
    indices = np.arange(1, len(data) + 1)

    return indices, cumulative_means.values


def plot_lvm_sample_analysis(
    lvm_path: Path,
    labbook_path: Path,
    results_dir: Path,
    block_size: int = 2500,
    window_size: int = 500,
) -> None:
    """Create block average and running average plots for F_Z and F_X from LVM sample"""

    # Set plot style
    set_plot_style(is_for_pdf_tex=False)

    # Process LVM file with labbook
    df = process_lvm_with_labbook(lvm_path, labbook_path)

    # Extract force data (already filtered for sideslip = 0)
    fz_data = df["F_Z"]
    fx_data = df["F_X"]

    # Calculate block averages
    block_indices_fz, block_means_fz = calculate_block_averages(fz_data, block_size)
    block_indices_fx, block_means_fx = calculate_block_averages(fx_data, block_size)

    # Calculate running averages
    run_indices_fz, run_means_fz = calculate_running_averages(fz_data, window_size)
    run_indices_fx, run_means_fx = calculate_running_averages(fx_data, window_size)

    # Create plots - single row with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Calculate x-positions for block averages (centered)
    block_x_fz = block_indices_fz * block_size - block_size / 2
    block_x_fx = block_indices_fx * block_size - block_size / 2

    # Create sample indices for raw data
    sample_indices = np.arange(1, len(fz_data) + 1)

    # Convert sample indices to time (assuming 2000 Hz sampling rate)
    sampling_rate = 2000  # Hz
    time_indices = sample_indices / sampling_rate
    time_block_x_fz = block_x_fz / sampling_rate
    time_block_x_fx = block_x_fx / sampling_rate
    time_run_indices_fz = run_indices_fz / sampling_rate
    time_run_indices_fx = run_indices_fx / sampling_rate

    # Calculate true averages
    fz_mean = fz_data.mean()
    fx_mean = fx_data.mean()

    # F_Z subplot (with lower alpha)
    axes[0].plot(
        time_indices,
        fz_data,
        color="black",
        linewidth=0.5,
        alpha=0.3,
        label="Raw data",
    )
    axes[0].axhline(
        y=fz_mean,
        color="black",
        linewidth=1.5,
        linestyle="-",
        alpha=1,
        label="True average",
    )
    axes[0].plot(
        time_run_indices_fz,
        run_means_fz,
        color="blue",
        linestyle=":",
        linewidth=3,
        alpha=1,
        label="Cumulative average",
    )
    axes[0].plot(
        time_block_x_fz,
        block_means_fz,
        color="red",
        marker="*",
        markersize=8,
        linewidth=0,
        alpha=1,
        label=f"Block average (size={block_size}samples)",
    )

    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel(r"$F_{\mathrm{z}}$ (N)")
    axes[0].set_xlim(0, 10)
    axes[0].grid(True, alpha=0.3)

    axes[0].set_ylim(680, 780)
    axes[1].set_ylim(40, 110)

    # Add top x-axis for sample numbers
    ax0_top = axes[0].twiny()
    ax0_top.set_xlim(0, 10 * sampling_rate)
    ax0_top.set_xlabel("Sample Number")

    # F_X subplot
    axes[1].plot(
        time_indices,
        fx_data,
        color="black",
        linewidth=0.5,
        alpha=0.5,
        # label="Raw data",
    )
    axes[1].axhline(
        y=fx_mean,
        color="black",
        linewidth=1.5,
        linestyle="-",
        alpha=1,
        # label="True average",
    )
    axes[1].plot(
        time_run_indices_fx,
        run_means_fx,
        color="blue",
        linestyle=":",
        linewidth=3,
        alpha=1,
        # label="Cumulative average",
    )
    axes[1].plot(
        time_block_x_fx,
        block_means_fx,
        color="red",
        marker="*",
        markersize=8,
        linewidth=0,
        alpha=1,
        # label=f"Block average (size={block_size})",
    )

    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel(r"$F_{\mathrm{x}}$ (N)")
    axes[1].set_xlim(0, 10)
    axes[1].grid(True, alpha=0.3)

    # Add top x-axis for sample numbers
    ax1_top = axes[1].twiny()
    ax1_top.set_xlim(0, 10 * sampling_rate)
    ax1_top.set_xlabel("Sample Number")

    # Add legend below the plot with 2 columns
    fig.legend(bbox_to_anchor=(0.5, 0.1), loc="center", ncol=2, frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)  # Make room for legend below

    # Save the figure
    filename = lvm_path.stem.replace("_unsteady", "")
    saving_pdf_and_pdf_tex(results_dir, f"lvm_sample_analysis_{filename}")

    # Print statistics
    print(f"\nStatistics for {filename}:")
    print(f"F_Z: mean = {fz_data.mean():.4f}, std = {fz_data.std():.4f}")
    print(f"F_X: mean = {fx_data.mean():.4f}, std = {fx_data.std():.4f}")
    print(f"Total samples: {len(df)}")
    print(f"Number of blocks: {len(block_means_fz)}")


def main(results_dir: Path, project_dir: Path) -> None:
    """Main function to run the LVM sample analysis"""
    # Hardcoded path to specific LVM file
    lvm_path = Path(
        "/home/jellepoland/ownCloud/phd/code/WES_load_wind_tunnel_measurements_TUDELFT_V3_LEI_KITE/data/normal/aoa_10/normal_aoa_10_vw_20_unsteady.lvm"
    )
    labbook_path = Path(project_dir) / "data" / "labbook.csv"

    # Check if files exist
    if not lvm_path.exists():
        raise FileNotFoundError(f"LVM file not found: {lvm_path}")
    if not labbook_path.exists():
        raise FileNotFoundError(f"Labbook file not found: {labbook_path}")

    # Create results directory if it doesn't exist
    results_dir.mkdir(exist_ok=True)

    # Run the analysis
    plot_lvm_sample_analysis(
        lvm_path=lvm_path,
        labbook_path=labbook_path,
        results_dir=results_dir,
        block_size=3000,  # Adjust as needed
        window_size=100,  # Adjust as needed
    )


if __name__ == "__main__":
    results_dir = Path(project_dir) / "results"
    main(results_dir, project_dir)
