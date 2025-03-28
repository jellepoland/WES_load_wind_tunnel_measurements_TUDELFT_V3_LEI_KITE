import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path
from load_balance_analysis.functions_utils import project_dir, save_latex_table
from plot_styling import plot_on_ax


def print_sensor_drift_values(results_path: Path, project_dir: Path) -> None:

    folder_dir = Path(project_dir) / "data" / "sensor_drift"
    # Load the data
    bod_april6 = np.mean(np.genfromtxt(Path(folder_dir) / "bod_april6.txt"), axis=0)
    bod_april8 = np.genfromtxt(Path(folder_dir) / "bod_april8.txt")
    bod_april9 = np.genfromtxt(Path(folder_dir) / "bod_april9.txt")

    eod_april5 = np.genfromtxt(Path(folder_dir) / "eod_april5.txt")
    eod_april6 = np.genfromtxt(Path(folder_dir) / "eod_april6.txt")
    eod_april8 = np.genfromtxt(Path(folder_dir) / "eod_april8.txt")

    tot = np.vstack(
        [eod_april5, bod_april6, eod_april6, bod_april8, eod_april8, bod_april9]
    )

    CL_list = []
    CD_list = []
    CS_list = []
    CMx_list = []
    CMy_list = []
    CMz_list = []
    for day in tot:
        CL_list.append(day[1])
        CD_list.append(day[2])
        CS_list.append(day[3])
        CMx_list.append(day[4])
        CMy_list.append(day[5])
        CMz_list.append(day[6])

    # Prepare the data for the table
    load_names = ["Fx [N]", "Fy [N]", "Fz [N]", "Mx [N/m]", "My [N/m]", "Mz [N/m]"]
    means = [
        np.mean(CL_list),
        np.mean(CD_list),
        np.mean(CS_list),
        np.mean(CMx_list),
        np.mean(CMy_list),
        np.mean(CMz_list),
    ]
    stds = [
        np.std(CL_list),
        np.std(CD_list),
        np.std(CS_list),
        np.std(CMx_list),
        np.std(CMy_list),
        np.std(CMz_list),
    ]

    # Create a DataFrame for the formatted output
    data = {
        "Load": load_names,
        "Mean": [f"{mean:.2f}" for mean in means],
        "std": [f"{std:.2f}" for std in stds],
    }

    df_table = pd.DataFrame(data)

    # Save the table as LaTeX
    save_latex_table(
        df_table, Path(project_dir) / "results" / "tables" / "sensor_drift.tex"
    )

    # Print the table to verify
    print(df_table.to_string(index=False))

    # Define the labels for the days
    # day_labels = [
    #     "Day 1 begin",
    #     "Day 1 end",
    #     "Day 2 begin",
    #     "Day 2 end",
    #     "Day 3 begin",
    #     "Day 3 end",
    # ]
    day_labels = [
        "00h",
        "12h",
        "24h",
        "36h",
        "48h",
        "60h",
    ]

    # Data lists for each component
    data_lists = [CL_list, CD_list, CS_list, CMx_list, CMy_list, CMz_list]
    component_labels = [
        r"$F_{\mathrm{x}}$ (N)",
        r"$F_{\mathrm{y}}$ (N)",
        r"$F_{\mathrm{z}}$ (N)",
        r"$M_{\mathrm{x}}$ (Nm)",
        r"$M_{\mathrm{y}}$ (Nm)",
        r"$M_{\mathrm{z}}$ (Nm)",
    ]

    # Create a 2x3 grid of plots
    fig, axes = plt.subplots(2, 3, figsize=(11, 5))
    axes = axes.flatten()  # Flatten the axes array for easier indexing

    for idx, (data, label) in enumerate(zip(data_lists, component_labels)):
        ax = axes[idx]

        # Calculate mean and standard deviation
        mean_val = np.mean(data)
        std_val = np.std(data)

        # Exclude x-labels for the top row
        is_with_x_label = idx >= 3

        # Use the plot_on_ax function for the data points
        plot_on_ax(
            ax=ax,
            x=day_labels,
            y=data,
            label="Value",
            linestyle="-",
            marker="o",
            markersize=8,
            x_label="Time (h)",
            y_label=label,
            is_with_x_label=is_with_x_label,
            is_with_x_ticks=is_with_x_label,
            is_with_y_label=True,
            is_with_grid=False,
        )

        # Plot the mean as a horizontal line
        ax.axhline(mean_val, color="black", linestyle="--", linewidth=1.5, label="Mean")

        # Add a shaded area for the standard deviation
        # ax.fill_between(
        #     range(len(day_labels)),
        #     mean_val - std_val,
        #     mean_val + std_val,
        #     color="gray",
        #     alpha=0.2,
        #     label=r"$\pm$1 Std Dev",
        # )

        # Add legend for the mean and std deviation
        if idx == 2:
            ax.legend(loc="lower left")

    # Adjust layout
    plt.tight_layout()
    plt.savefig(Path(results_path) / "sensor_drift.pdf")


def main(results_path: Path, project_dir: Path) -> None:
    print_sensor_drift_values(results_path, project_dir)
