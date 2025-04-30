import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path
from load_balance_analysis.functions_utils import (
    saving_pdf_and_pdf_tex,
    x_axis_labels,
    y_axis_labels,
    project_dir,
)
from load_balance_analysis.functions_statistics import (
    calculate_confidence_interval,
    block_bootstrap_confidence_interval,
    hac_newey_west_confidence_interval,
)


## PLOTS STANDARD DEVIATION
def generate_boxplots_vs_aoa(
    combined_df: pd.DataFrame,
    vw: str,
    sideslip: float,
    results_path: str,
    figsize,
    fontsize,
    columns,
    y_labels,
    subplot_titles,
) -> None:
    """
    Generate and save boxplots for different load types versus angle of attack.

    Args:
    - combined_df (pd.DataFrame): DataFrame containing all processed data.
    - vw (str): Wind speed value as a string.
    - sideslip (int): Sideslip angle.
    - titlespeeds (list): List of wind speed titles for labeling plots.
    - j (int): Index of the current wind speed in the list.
    """

    if len(columns) == 6:
        fig, axs = plt.subplots(2, 3, figsize=figsize)
    elif len(columns) == 3:
        fig, axs = plt.subplots(1, 3, figsize=figsize)
    elif len(columns) == 2:
        fig, axs = plt.subplots(1, 2, figsize=figsize)

    cmap = plt.get_cmap("rainbow")
    colors = cmap(np.linspace(0, 1, len(columns)))

    axs = axs.flatten()

    for i, load in enumerate(columns):
        unique_aoa = sorted(combined_df["aoa_kite"].unique())
        box_data = [
            combined_df[combined_df["aoa_kite"] == aoa][load] for aoa in unique_aoa
        ]
        positions = unique_aoa

        means = [data.mean() for data in box_data]

        ### ERROR BARS
        CI_value = 0.99
        alpha_CI = 1 - CI_value
        # standard deviation
        std = [data.std() for data in box_data]

        axs[i].errorbar(
            positions, means, yerr=std, fmt="o", color="black", capsize=5, markersize=1
        )
        axs[i].plot(positions, means, marker="o", linestyle="--", color="black")

        axs[i].set_ylabel(y_axis_labels[y_labels[i]])  # , fontsize=fontsize)
        axs[i].set_xticks(range(-15, 26, 5))
        axs[i].set_xticklabels(range(-15, 26, 5))
        axs[i].set_xlim(-15, 25)
        axs[i].grid()

    for ax in axs:
        ax.set_xlabel(x_axis_labels["alpha"])  # , fontsize=fontsize)

    plt.tight_layout()

    filename = f"boxplot_alpha_sweep_at_fixed_beta_{sideslip:.2f}_vw_{vw}_STD"  # CI_NEWEYWEST_{CI_value}"
    saving_pdf_and_pdf_tex(results_path, filename)


def generate_boxplots_vs_sideslip(
    combined_df: pd.DataFrame,
    vw: str,
    aoa: float,
    results_path: str,
    figsize,
    fontsize,
    columns,
    y_labels,
    subplot_titles,
) -> None:
    """
    Generate and save boxplots for different load types versus sideslip angle.

    Args:
    - combined_df (pd.DataFrame): DataFrame containing all processed data.
    - vw (str): Wind speed value as a string.
    - aoa (float): Angle of attack.
    - titlespeeds (list): List of wind speed titles for labeling plots.
    - j (int): Index of the current wind speed in the list.
    """
    columns = ["C_D", "C_S", "C_L", "C_roll", "C_pitch", "C_yaw"]
    y_labels = ["CD", "CS", "CL", "CMx", "CMy", "CMz"]
    subplot_titles = [
        "Drag coefficient",
        "Side force coefficient",
        "Lift coefficient",
        "Rolling moment coefficient",
        "Pitching moment coefficient",
        "Yawing moment coefficient",
    ]

    unique_sideslip = sorted(combined_df["sideslip"].unique())

    if (
        len(unique_sideslip) > 1
    ):  # Only create plots if there is more than one unique sideslip value

        cmap = plt.get_cmap("rainbow")
        colors = cmap(np.linspace(0, 1, len(columns)))

        fig, axs = plt.subplots(2, 3, figsize=figsize)
        axs = axs.flatten()

        for i, load in enumerate(columns):
            box_data = [
                combined_df[combined_df["sideslip"] == sideslip][load]
                for sideslip in unique_sideslip
            ]
            positions = unique_sideslip

            bplot = axs[i].boxplot(
                box_data,
                positions=positions,
                widths=0.6,
                showfliers=False,
                patch_artist=True,
            )
            for patch, color in zip(
                bplot["boxes"], cmap(np.linspace(0, 1, len(unique_sideslip)))
            ):
                patch.set_facecolor(color)

            means = [data.mean() for data in box_data]
            axs[i].plot(positions, means, marker="o", linestyle="--", color="black")
            # axs[i].set_title(f"{subplot_titles[i]}")
            axs[i].set_ylabel(rf"${y_labels[i]}$ (-)")  # , fontsize=fontsize)
            axs[i].set_xticks(
                np.arange(
                    np.floor(min(unique_sideslip)), np.ceil(max(unique_sideslip)) + 1, 4
                )
            )
            axs[i].set_xticklabels(
                np.arange(
                    np.floor(min(unique_sideslip)), np.ceil(max(unique_sideslip)) + 1, 4
                ).astype(int)
            )
            axs[i].grid()

        for ax in axs:
            ax.set_xlabel(x_axis_labels["beta"])  # , fontsize=fontsize)

        plt.tight_layout()
        # output_dir = f"plots_unsteady_final/vw_{vw}/beta"
        # os.makedirs(output_dir, exist_ok=True)
        plot_filename = (
            Path(results_path)
            / f"boxplot_beta_sweep_at_fixed_aoa_{np.around(aoa,2)}_vw_{vw}.pdf"
        )
        plt.savefig(plot_filename)
        plt.close()


def main(results_path, project_dir):

    # Define folder path, wind speeds, sideslip angles, and titles
    normal_folder_path = Path(project_dir) / "Mark_van_Spronsen" / "normal"
    # all
    plot_speeds = ["05", "10", "15", "20", "25"]
    betas_to_be_plotted = [
        -20,
        -14,
        -12,
        -10,
        -8,
        -6,
        -4,
        -2,
        0,
        2,
        4,
        6,
        8,
        12,
        14,
        20,
    ]
    alphas_to_be_plotted = [
        -12.65,
        -7.15,
        -3.0,
        -2.25,
        2.35,
        4.75,
        6.75,
        8.8,
        10.95,
        11.95,
        12.8,
        14.0,
        15.75,
        17.85,
        19.75,
        22.55,
        24.0,
    ]

    # folder normal_csv
    normal_csv_dir = Path(project_dir) / "processed_data" / "normal_csv"

    # selection
    plot_speeds = ["20"]
    betas_to_be_plotted = [0]
    alphas_to_be_plotted = [6.75, 11.95]
    fontsize = 18

    # Loop through each wind speed
    for j, vw in enumerate(plot_speeds):
        for sideslip in betas_to_be_plotted:

            ## NEW
            path_to_csv = Path(normal_csv_dir) / f"beta_{sideslip}" / f"vw_{vw}.csv"
            combined_df = pd.read_csv(path_to_csv)

            figsize = (16, 8)
            # columns = ["C_L", "C_D", "C_pitch"]
            # y_labels = ["C_L", "C_D", "C_{M,x}"]
            columns = ["C_L", "C_D"]
            y_labels = ["CL", "CD"]
            subplot_titles = [
                "Lift coefficient",
                "Drag Coefficient",
                # "Pitch moment coefficient",
            ]

            generate_boxplots_vs_aoa(
                combined_df[combined_df["sideslip"] == sideslip],
                vw,
                sideslip,
                results_path,
                figsize,
                fontsize,
                columns,
                y_labels,
                subplot_titles,
            )  # Generate and save boxplots vs AoA

            ### we only need the boxplots over an angle of attack sweep
            ###TODO: code below is not being used...

        unique_aoas = combined_df["aoa_kite"].unique()

        ## NEW
        alphas_to_be_plotted = [6.8, 11.9]

        for alpha in alphas_to_be_plotted:
            ## NEW
            path_to_csv = Path(normal_csv_dir) / f"alpha_{alpha}" / f"vw_{vw}.csv"
            combined_df = pd.read_csv(path_to_csv)

            figsize = (16, 12)
            columns = ["C_L", "C_D", "C_S", "C_roll", "C_pitch", "C_yaw"]
            y_labels = ["C_L", "C_D", "C_S", "C_{M,x}", "C_{M,y}", "C_{M,z}"]
            subplot_titles = [
                "Lift coefficient",
                "Drag Coefficient",
                "Side Force coefficient",
                "Pitch moment coefficient",
                "Roll moment coefficient",
                "Yaw moment coefficient",
            ]
            generate_boxplots_vs_sideslip(
                combined_df,
                vw,
                alpha,
                results_path,
                figsize,
                fontsize,
                columns,
                y_labels,
                subplot_titles,
            )  # Generate and save boxplots vs sideslip


if __name__ == "__main__":
    results_path = Path(project_dir) / "results"
    main(results_path, project_dir)
