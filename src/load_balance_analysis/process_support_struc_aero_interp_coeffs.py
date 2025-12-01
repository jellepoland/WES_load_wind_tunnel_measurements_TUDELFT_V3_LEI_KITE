import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from load_balance_analysis.functions_utils import (
    project_dir,
    reduce_df_by_parameter_mean_and_std,
)
from load_balance_analysis.functions_processing import processing_raw_lvm_data_into_csv


def process_and_merge_without_csv(folder_dir: Path, save_path: Path) -> None:
    # Initialize lists to store all data
    all_data = []

    # Get all alpha folders
    alpha_folders = [f for f in folder_dir.glob("alpha_*")]

    # Process each alpha folder
    for alpha_folder in alpha_folders:
        # Extract alpha value from folder name
        alpha_str = alpha_folder.name.split("_")[1]
        alpha = float(alpha_str)

        # Process each wind velocity file
        for vw_file in alpha_folder.glob("vw_*.csv"):
            # Extract vw value from filename
            vw_str = vw_file.stem.split("_")[1]
            vw = float(vw_str)

            # Read and process the CSV file
            df = pd.read_csv(vw_file)

            # Reduce the dataframe using your existing function
            reduced_df = reduce_df_by_parameter_mean_and_std(
                df, parameter="sideslip", is_support=True
            )

            # Add alpha column
            reduced_df["aoa"] = alpha

            # Append to all_data
            all_data.append(reduced_df)

    # Combine all processed data
    processed_df = pd.concat(all_data, ignore_index=True)

    # Saving
    processed_df.to_csv(save_path, index=False)


def interpolation_1_line(processed_df: pd.DataFrame, save_path: Path) -> None:
    """
    Perform interpolation on data organized in folders by alpha values.

    Args:
        save_path: Path where to save the interpolation coefficients
    """
    # Prepare data for interpolation
    output_columns = ["C_L", "C_D", "C_S", "C_roll", "C_pitch", "C_yaw"]

    # Get unique values
    betalist = sorted(processed_df["sideslip"].unique())
    vwlist = sorted(processed_df["vw"].unique())

    def pol_inter(x, y, order):
        """Polynomial interpolation function"""
        V = np.vander(x, order + 1)

        if order == 2:
            coefficients = np.linalg.solve(V, y)
        elif order == 1:
            coeff, res, rank, s = np.linalg.lstsq(V, y, rcond=None)
            b, c = coeff
            a = 0
            coefficients = np.array([a, b, c])

        return coefficients

    # Create empty dataframe for interpolation coefficients
    c_interp = pd.DataFrame(columns=["channel", "a", "b", "c", "sideslip", "vw"])
    row = 0

    # Set interpolation order
    order = 1  # 1 for linear, 2 for quadratic

    # Perform interpolation
    for beta in betalist:
        for vw in vwlist:
            for channel in output_columns:
                # Filter data for current beta and vw
                mask = (processed_df["sideslip"] == beta) & (processed_df["vw"] == vw)
                current_data = processed_df[mask].sort_values("aoa")

                if len(current_data) < 2:
                    print(
                        f"Warning: Insufficient data points for beta={beta}, vw={vw}, channel={channel}"
                    )
                    continue

                x_val = current_data["aoa"].values
                y_val = current_data[channel].values

                try:
                    a, b, c = pol_inter(x_val, y_val, order)

                    # Append to dataframe
                    c_interp.loc[row] = [channel, a, b, c, beta, vw]
                    row += 1
                except np.linalg.LinAlgError:
                    print(
                        f"Warning: Could not compute interpolation for beta={beta}, vw={vw}, channel={channel}"
                    )
                    continue

    # Save interpolation coefficients
    c_interp.to_csv(save_path, index=False)


def plot_interpolation_fit_1_line(
    processed_df: pd.DataFrame,
    df_interpolation: pd.DataFrame,
    beta: float,
    save_fig_path=Path,
) -> None:
    """
    Plot raw data points and interpolation fits for aerodynamic coefficients.
    The interpolation is done as a function of alpha (angle of attack) for fixed
    beta (sideslip) and plots all available wind velocities.

    Args:
        processed_df: DataFrame containing the raw data
        df_interpolation: DataFrame containing the interpolation coefficients
        beta: float, sideslip angle value to plot for
    """
    # Print debug information about the data
    print(f"Available alpha values: {sorted(processed_df['aoa'].unique())}")
    print(f"Available beta values: {sorted(processed_df['sideslip'].unique())}")
    print(f"Available vw values: {sorted(processed_df['vw'].unique())}")

    # Create color map for different wind velocities
    colors = ["blue", "orange", "green", "red", "purple"]
    vw_list = np.array([5, 10, 15, 20, 25])
    vw_colors = dict(zip(vw_list, colors))

    # Create figure with 2 rows, 3 columns, and extra space at bottom for legend
    fig = plt.figure(figsize=(15, 11))  # Made figure slightly taller

    # Create GridSpec to accommodate plots and legend
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.2])

    # Create axes for the plots
    axes = []
    for i in range(2):
        for j in range(3):
            axes.append(fig.add_subplot(gs[i, j]))

    # Flatten axes list
    axes = np.array(axes)

    # List of output columns to plot
    output_columns = ["C_L", "C_D", "C_S", "C_roll", "C_pitch", "C_yaw"]

    # Store first legend handles and labels
    legend_handles = []
    legend_labels = []

    for idx, channel in enumerate(output_columns):
        ax = axes[idx]

        for vw in vw_list:
            # Filter raw data for the specified conditions
            mask = (processed_df["vw"] == vw) & (processed_df["sideslip"] == beta)
            filtered_data = processed_df[mask].copy()

            if filtered_data.empty:
                print(f"Warning: No data points found for vw={vw}, beta={beta}")
                continue

            # Create smooth alpha values for the interpolation curve
            alpha_values = sorted(filtered_data["aoa"].unique())
            alpha_smooth = np.linspace(min(alpha_values), max(alpha_values), 100)

            # Plot raw data points
            scatter = ax.scatter(
                filtered_data["aoa"],
                filtered_data[channel],
                color=vw_colors[vw],
                alpha=0.6,
                marker="o",
            )

            # Get interpolation coefficients for this channel and conditions
            coeff_mask = (
                (df_interpolation["channel"] == channel)
                & (df_interpolation["vw"] == vw)
                & (df_interpolation["sideslip"] == beta)
            )

            # Check if we have matching coefficients
            if not any(coeff_mask):
                print(
                    f"Warning: No coefficients found for {channel} at vw={vw}, beta={beta}"
                )
                continue

            coeffs = df_interpolation[coeff_mask].iloc[0]

            # Calculate interpolated values
            y_smooth = (
                coeffs["a"] * alpha_smooth**2 + coeffs["b"] * alpha_smooth + coeffs["c"]
            )

            # Plot interpolation curve
            line = ax.plot(
                alpha_smooth,
                y_smooth,
                "--",
                color=vw_colors[vw],
            )

            # Store legend handles and labels only from the first subplot
            if idx == 0:
                legend_handles.extend([scatter, line[0]])
                legend_labels.extend(
                    [f"Raw Data (vw={vw})", f"Interpolation (vw={vw})"]
                )

        # Set labels and title
        ax.set_xlabel("α (angle of attack) [deg]")
        ax.set_ylabel(channel)
        ax.set_title(f"{channel} vs α")
        ax.grid(True)

    # Create a new axis for the legend at the bottom
    legend_ax = fig.add_subplot(gs[2, :])
    legend_ax.axis("off")  # Hide the axis

    # Add the legend to the bottom
    legend_ax.legend(
        legend_handles,
        legend_labels,
        loc="center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.5),
        fontsize="small",
    )

    # Add overall title
    fig.suptitle(
        f"Aerodynamic Coefficients for β = {beta}°\n"
        "Comparison across wind velocities",
        y=0.98,
    )

    # Adjust layout
    plt.tight_layout()
    # plt.show()
    plt.savefig(save_fig_path)


def interpolation(processed_df: pd.DataFrame, save_path: Path) -> None:
    """
    Perform interpolation on data organized in folders by alpha values.

    Args:
        save_path: Path where to save the interpolation coefficients
    """
    # Prepare data for interpolation
    output_columns = ["C_L", "C_D", "C_S", "C_roll", "C_pitch", "C_yaw"]

    # Get unique values
    betalist = sorted(processed_df["sideslip"].unique())
    vwlist = sorted(processed_df["vw"].unique())

    def linear_fit(x, y):
        """Linear least-squares fit"""
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[
            0
        ]  # Solve for slope (m) and intercept (c)
        return m, c

    # Create empty dataframe for interpolation coefficients
    max_degree = 3  # maximum polynomial degree we will store (0..3)
    coef_columns = [f"coef_{i}" for i in range(max_degree + 1)]
    c_interp = pd.DataFrame(
        columns=["channel", "degree", *coef_columns, "sideslip", "vw"]
    )
    row = 0

    # Perform interpolation
    for beta in betalist:
        for vw in vwlist:
            for channel in output_columns:
                # Filter data for current beta and vw
                mask = (processed_df["sideslip"] == beta) & (processed_df["vw"] == vw)
                current_data = processed_df[mask].sort_values("aoa")

                if len(current_data) < 3:
                    print(
                        f"Warning: Insufficient data points for beta={beta}, vw={vw}, channel={channel}"
                    )
                    continue

                x_val = current_data["aoa"].values
                y_val = current_data[channel].values

                degree = min(2, len(x_val) - 1)
                if degree < 1:
                    print(
                        f"Warning: Insufficient data spread for beta={beta}, vw={vw}, channel={channel}"
                    )
                    continue

                coeffs_desc = np.polyfit(x_val, y_val, degree)
                coeffs_asc = coeffs_desc[::-1]  # constant first
                stored_coeffs = [0.0] * (max_degree + 1)
                for idx, coef in enumerate(coeffs_asc):
                    stored_coeffs[idx] = coef

                c_interp.loc[row] = [
                    channel,
                    degree,
                    *stored_coeffs,
                    beta,
                    vw,
                ]
                row += 1

    # Save interpolation coefficients
    c_interp.to_csv(save_path, index=False)


def plot_interpolation_fit(
    processed_df: pd.DataFrame,
    df_interpolation: pd.DataFrame,
    beta: float,
    save_fig_path=Path,
) -> None:
    """
    Plot raw data points and interpolation fits for aerodynamic coefficients.
    The interpolation is done as a function of alpha (angle of attack) for fixed
    beta (sideslip) and plots all available wind velocities using two linear fits.

    Args:
        processed_df: DataFrame containing the raw data
        df_interpolation: DataFrame containing the interpolation coefficients
        beta: float, sideslip angle value to plot for
    """
    # Print debug information about the data
    print(f"Available alpha values: {sorted(processed_df['aoa'].unique())}")
    print(f"Available beta values: {sorted(processed_df['sideslip'].unique())}")
    print(f"Available vw values: {sorted(processed_df['vw'].unique())}")

    # Create color map for different wind velocities
    colors = ["blue", "orange", "green", "red", "purple"]
    vw_list = np.array([5, 10, 15, 20, 25])
    vw_colors = dict(zip(vw_list, colors))

    # Create figure with 2 rows, 3 columns, and extra space at bottom for legend
    fig = plt.figure(figsize=(15, 11))  # Made figure slightly taller

    # Create GridSpec to accommodate plots and legend
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.2])

    # Create axes for the plots
    axes = []
    for i in range(2):
        for j in range(3):
            axes.append(fig.add_subplot(gs[i, j]))

    # Flatten axes list
    axes = np.array(axes)

    # List of output columns to plot
    output_columns = ["C_L", "C_D", "C_S", "C_roll", "C_pitch", "C_yaw"]

    # Store first legend handles and labels
    legend_handles = []
    legend_labels = []

    for idx, channel in enumerate(output_columns):
        ax = axes[idx]

        for vw in vw_list:
            # Filter raw data for the specified conditions
            mask = (processed_df["vw"] == vw) & (processed_df["sideslip"] == beta)
            filtered_data = processed_df[mask].copy()

            if filtered_data.empty:
                print(f"Warning: No data points found for vw={vw}, beta={beta}")
                continue

            # Create smooth alpha values for the interpolation curve
            alpha_values = sorted(filtered_data["aoa"].unique())
            alpha_smooth = np.linspace(min(alpha_values), max(alpha_values), 100)

            # Plot raw data points
            scatter = ax.scatter(
                filtered_data["aoa"],
                filtered_data[channel],
                color=vw_colors[vw],
                alpha=0.6,
                marker="o",
            )

            # Get interpolation coefficients for this channel and conditions
            coeff_mask = (
                (df_interpolation["channel"] == channel)
                & (df_interpolation["vw"] == vw)
                & (df_interpolation["sideslip"] == beta)
            )

            # Check if we have matching coefficients
            if not any(coeff_mask):
                print(
                    f"Warning: No coefficients found for {channel} at vw={vw}, beta={beta}"
                )
                continue

            coeffs = df_interpolation[coeff_mask].iloc[0]
            degree = int(coeffs["degree"])
            coef_vals = [
                float(coeffs[f"coef_{i}"]) for i in range(degree + 1)
            ]
            y_smooth = np.polyval(coef_vals[::-1], alpha_smooth)

            # Plot the polynomial fit
            line = ax.plot(alpha_smooth, y_smooth, "--", color=vw_colors[vw])

            # Store legend handles and labels only from the first subplot
            if idx == 0:
                legend_handles.extend([scatter, line[0]])
                legend_labels.extend(
                    [
                        f"Raw Data (vw={vw})",
                        f"Polynomial Fit (vw={vw})",
                    ]
                )

        # Set labels and title
        ax.set_xlabel("α (angle of attack) [deg]")
        ax.set_ylabel(channel)
        ax.set_title(f"{channel} vs α")
        ax.grid(True)

    # Create a new axis for the legend at the bottom
    legend_ax = fig.add_subplot(gs[2, :])
    legend_ax.axis("off")  # Hide the axis

    # Add the legend to the bottom
    legend_ax.legend(
        legend_handles,
        legend_labels,
        loc="center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.5),
        fontsize="small",
    )

    # Add overall title
    fig.suptitle(
        f"Aerodynamic Coefficients for β = {beta}°\n"
        "Comparison across wind velocities",
        y=0.98,
    )

    # Adjust layout
    plt.tight_layout()
    # plt.show()

    plt.savefig(save_fig_path)


def main():

    # processing all the folders for the without case
    support_struc_aero_interp_coeffs_path = None
    is_kite = False
    is_zigzag = False
    without_csv_dir = Path(project_dir) / "processed_data" / "without_csv"
    print(f"\n Processing all the folders")
    for folder in os.listdir(without_csv_dir):
        if "alpha" in folder:
            folder_dir = Path(without_csv_dir) / folder
            processing_raw_lvm_data_into_csv(
                folder_dir,
                is_kite,
                is_zigzag,
                support_struc_aero_interp_coeffs_path,
            )

    ## Process and merge
    save_without_processed_and_merged_path = (
        Path(project_dir)
        / "processed_data"
        / "without_csv"
        / "without_processed_and_merged.csv"
    )
    process_and_merge_without_csv(
        without_csv_dir, save_without_processed_and_merged_path
    )

    ### 1-line interpolation
    # Interpolate
    processed_df = pd.read_csv(save_without_processed_and_merged_path)
    save_interpolation_path = (
        Path(project_dir) / "processed_data" / "1_line_interpolation_coefficients.csv"
    )
    interpolation_1_line(processed_df, save_interpolation_path)

    # Plotting the fit 1-line
    df_interpolation = pd.read_csv(save_interpolation_path)
    # Load your saved coefficients
    save_fig_path = (
        Path(project_dir)
        / "processed_data"
        / "without_csv"
        / "interpolation_fit_1_linear_line.png"
    )
    plot_interpolation_fit_1_line(
        processed_df=processed_df,
        df_interpolation=df_interpolation,
        beta=0,  # sideslip angle
        save_fig_path=save_fig_path,
    )

    ### 2-line interpolation
    # Interpolate
    processed_df = pd.read_csv(save_without_processed_and_merged_path)
    save_interpolation_path = (
        Path(project_dir) / "processed_data" / "interpolation_coefficients.csv"
    )
    interpolation(processed_df, save_interpolation_path)

    # Plotting the fit
    df_interpolation = pd.read_csv(save_interpolation_path)
    # Load your saved coefficients
    save_fig_path = (
        Path(project_dir)
        / "processed_data"
        / "without_csv"
        / "interpolation_fit_2_linear_line.png"
    )
    plot_interpolation_fit(
        processed_df=processed_df,
        df_interpolation=df_interpolation,
        beta=0,  # sideslip angle
        save_fig_path=save_fig_path,
    )


if __name__ == "__main__":
    main()
