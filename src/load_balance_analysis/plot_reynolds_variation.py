import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from pathlib import Path
from load_balance_analysis.functions_utils import (
    saving_pdf_and_pdf_tex,
    x_axis_labels,
    y_axis_labels,
    reduce_df_by_parameter_mean_and_std,
    project_dir,
    apply_angle_wind_tunnel_corrections_to_df,
)
from plot_styling import plot_on_ax


def plotting_CL_CD_CS_Pitch_Roll_Yaw_vs_alpha_reynolds_sweep(
    results_path: str,
    stats_all: pd.DataFrame,
    betas_to_be_plotted: str,
    plot_speeds: list,
    columns: list,
    y_labels: list,
    subplot_titles: list,
):

    # # set which wind speeds to plot
    # plot_speeds = [5, 10, 15, 20, 25]

    if len(plot_speeds) < 5:
        subfolder = "vw="
        for i, v in enumerate(plot_speeds):
            if i == 0:
                subfolder = subfolder + f"{v}"
            else:
                subfolder = subfolder + f"+{v}"
    else:
        subfolder = "all_vw"

    # # Create a new folder to save the plots
    # os.makedirs(foldername + "/alpha", exist_ok=True)

    # sort everything for plotting correctly
    stats_plotvsalpha = stats_all.sort_values(by="aoa_kite")

    # Group the data by sideslip
    grouped = stats_plotvsalpha.groupby("sideslip")

    # Loop through each sideslip value
    for sideslip, group in grouped:

        if sideslip in betas_to_be_plotted:
            # Create a subplot with 4 rows and 3 columns
            fig, axs = plt.subplots(2, 3, figsize=(15, 10))

            # Flatten the subplot array for easier indexing
            axs = axs.flatten()

            linestyles = {
                "5": "--",
                "10": "--",
                "15": "-",
                "20": "-",
                "25": "-",
            }
            markers = {
                "5": "s",
                "10": "X",
                "15": "d",
                "20": "o",
                "25": "*",
            }

            for i, column in enumerate(columns):

                if column in ["C_L", "C_D", "C_S"]:
                    is_with_x_ticks = False
                    is_with_x_label = False
                else:
                    is_with_x_ticks = True
                    is_with_x_label = True

                # Plot each distinct value in the vw column (excluding vw=0 and vw=5)
                for vw, vw_group in group.groupby("vw"):
                    if vw in plot_speeds:
                        Re = np.around((vw_group["Rey"].mean()) / 1e5, 1)

                        if vw == 5:
                            Re = 1.3
                        elif vw == 10:
                            Re = 2.5
                        elif vw == 15:
                            Re = 3.8
                        elif vw == 20:
                            Re = 5
                        elif vw == 25:
                            Re = 6.1

                        vw_group = apply_angle_wind_tunnel_corrections_to_df(vw_group)
                        axs[i].set_xlim(-13, 24)

                        plot_on_ax(
                            axs[i],
                            vw_group["aoa_kite"],
                            vw_group[column],
                            linestyle=linestyles[str(int(vw))],
                            marker=markers[str(int(vw))],
                            label=rf"Re = {Re} $\times$ $10^5$",
                            is_with_x_ticks=is_with_x_ticks,
                            is_with_x_label=is_with_x_label,
                            x_label=x_axis_labels["alpha"],
                            y_label=y_axis_labels[y_labels[i]],
                        )

                if i == 0:
                    axs[i].legend()
            plt.tight_layout()
            filename = f"re_variation_alpha_sweep_at_fixed_beta_{sideslip:.2f}"
            saving_pdf_and_pdf_tex(results_path, filename)


def plotting_CL_CD_CS_Pitch_Roll_Yaw_vs_beta_reynolds_sweep(
    results_path: str,
    stats_all: pd.DataFrame,
    alphas_to_be_plotted: list,
    plot_speeds: list,
    columns: list,
    y_labels: list,
    subplot_titles: list,
):
    # # set which wind speeds to plot
    # plot_speeds = [5, 10, 15, 20, 25]

    if len(plot_speeds) < 5:
        subfolder = "vw="
        for i, v in enumerate(plot_speeds):
            if i == 0:
                subfolder = subfolder + f"{v}"
            else:
                subfolder = subfolder + f"+{v}"
    else:
        subfolder = "all_vw"

    # sort everything for plotting correctly
    stats_plotvsbeta = stats_all.sort_values(by="sideslip")

    # Group the data by sideslip
    grouped = stats_plotvsbeta.groupby("aoa_kite")

    # Loop through each sideslip value
    for alpha, group in grouped:

        if alpha in alphas_to_be_plotted:
            # only plot if there is more than one entry per wind speed
            entries = len(group["vw"])
            if entries > 5:
                # Create a subplot with 4 rows and 3 columns
                fig, axs = plt.subplots(2, 3, figsize=(15, 10))

                # Flatten the subplot array for easier indexing
                axs = axs.flatten()

                linestyles = {
                    "5": "--",
                    "10": "--",
                    "15": "--",
                    "20": "--",
                    "25": "--",
                }
                markers = {
                    "5": "s",
                    "10": "X",
                    "15": "d",
                    "20": "o",
                    "25": "*",
                }
                for i, column in enumerate(columns):

                    if column in ["C_L", "C_D", "C_S"]:
                        is_with_x_ticks = False
                        is_with_x_label = False
                    else:
                        is_with_x_ticks = True
                        is_with_x_label = True

                    # Plot each distinct value in the vw column (excluding vw=0 and vw=5)
                    for vw, vw_group in group.groupby("vw"):
                        if vw in plot_speeds:
                            Re = np.around((vw_group["Rey"].mean()) / 1e5, 1)
                            if vw == 5:
                                Re = 1.3
                            elif vw == 10:
                                Re = 2.5
                            elif vw == 15:
                                Re = 3.8
                            elif vw == 20:
                                Re = 5
                            elif vw == 25:
                                Re = 6.1

                            vw_group = apply_angle_wind_tunnel_corrections_to_df(
                                vw_group
                            )
                            print(f'column: {column}, alpha: {vw_group["aoa_kite"]}')
                            plot_on_ax(
                                axs[i],
                                vw_group["sideslip"],
                                vw_group[column],
                                linestyle=linestyles[str(int(vw))],
                                marker=markers[str(int(vw))],
                                label=rf"Re = {Re} $\times$ $10^5$",
                                is_with_x_ticks=is_with_x_ticks,
                                is_with_x_label=is_with_x_label,
                                x_label=x_axis_labels["beta"],
                                y_label=y_axis_labels[y_labels[i]],
                            )
                            axs[i].set_xlim(-20, 20)

                            pos_values, neg_values = [], []

                            for j, column_j in enumerate(vw_group[column].values):
                                beta = vw_group["sideslip"].values[j]

                                if beta > 1:
                                    pos_values.append(column_j)
                                elif beta < 1:
                                    neg_values.append(column_j)
                            diff = []
                            for idx, (pos, neg) in enumerate(
                                zip(pos_values, neg_values[::-1])
                            ):
                                if idx > 3:
                                    diff.append(np.abs(1 - np.abs(pos / neg)))

                    if i == 1:
                        axs[i].legend(loc="upper center")
                plt.tight_layout()
                filename = f"re_variation_beta_sweep_at_fixed_alpha_{alpha:.2f}"
                saving_pdf_and_pdf_tex(results_path, filename)


def plotting_CL_CD_CS_Pitch_Roll_Yaw_vs_beta_total_kite_support(
    results_path: str,
    stats_all: pd.DataFrame,
    alphas_to_be_plotted: list,
    plot_speeds: list,  # Not used since only vw=20 is plotted
    columns: list,
    y_labels: list,
    subplot_titles: list,  # Not used since titles are removed
):
    """
    Plots, for each alpha in alphas_to_be_plotted, the 'kite' contribution (e.g. C_L),
    the 'support' contribution (C_L_s), and their total (C_L + C_L_s),
    across sideslip (beta) for a fixed wind speed of 20 m/s.

    Parameters
    ----------
    results_path : str
        Path to directory where figures will be saved.
    stats_all : pd.DataFrame
        DataFrame containing columns like [vw, sideslip, aoa_kite, C_L, C_L_s, C_D, C_D_s, ...].
    alphas_to_be_plotted : list
        Which alpha (aoa_kite) values to plot.
    plot_speeds : list
        Not used in this function since only vw=20 is plotted.
    columns : list
        Which aerodynamic coefficients to plot, e.g. ["C_L", "C_D", "C_S", "C_roll", "C_pitch", "C_yaw"].
    y_labels : list
        Y-axis labels, same length as columns.
    subplot_titles : list
        Titles for each subplot, same length as columns (not used).
    """

    # Ensure only positive beta values are plotted
    stats_plotvsbeta = stats_all[stats_all["sideslip"] > -1].sort_values(by="sideslip")

    # Group by alpha
    grouped = stats_plotvsbeta.groupby("aoa_kite")

    # Define fixed wind speed
    fixed_vw = 20

    # Define styles
    styles = {
        "kite": {"color": "red", "marker": "o", "linestyle": "-", "label": "Kite"},
        "support": {
            "color": "blue",
            "marker": "s",
            "linestyle": "-",
            "label": "Support",
        },
        "total": {"color": "black", "marker": "*", "linestyle": "-", "label": "Total"},
    }

    for alpha_value, group in grouped:
        if alpha_value in alphas_to_be_plotted:
            # Filter for fixed wind speed
            vw_group = group[group["vw"] == fixed_vw]

            # Check if there is sufficient data
            if len(vw_group) < 3:
                print(
                    f"Skipping alpha={alpha_value} due to insufficient data (n={len(vw_group)})"
                )
                continue

            # Create a subplot with 2 rows and 3 columns
            fig, axs = plt.subplots(2, 3, figsize=(14, 8))
            axs = axs.flatten()

            for i, column in enumerate(columns):
                ax = axs[i]

                # Extract sideslip and ensure positive values
                x_vals = vw_group["sideslip"].values

                # 'Kite' is the nominal column
                y_kite = vw_group[column].values

                # 'Support' is the subscript _s column
                support_col = column + "_s"
                if support_col not in vw_group.columns:
                    # If no support column, assume zero and warn
                    y_support = np.zeros_like(y_kite)
                    print(f"Warning: No support column for {column}, assuming zero.")
                else:
                    y_support = vw_group[support_col].values

                # 'Total' = sum of the two
                y_total = y_kite + y_support

                # Plot 'kite'
                ax.plot(
                    x_vals,
                    y_kite,
                    color=styles["kite"]["color"],
                    marker=styles["kite"]["marker"],
                    linestyle=styles["kite"]["linestyle"],
                    label=styles["kite"]["label"],
                )

                # Plot 'support'
                ax.plot(
                    x_vals,
                    y_support,
                    color=styles["support"]["color"],
                    marker=styles["support"]["marker"],
                    linestyle=styles["support"]["linestyle"],
                    label=styles["support"]["label"],
                )

                # Plot 'total'
                ax.plot(
                    x_vals,
                    y_total,
                    color=styles["total"]["color"],
                    marker=styles["total"]["marker"],
                    linestyle=styles["total"]["linestyle"],
                    label=styles["total"]["label"],
                )

                # Set x and y labels
                ax.set_xlabel(r"$\beta$ [°]")  # , fontsize=12)
                ax.set_ylabel(y_axis_labels[y_labels[i]])  # , fontsize=12)

                # Enable grid
                ax.grid(True)

            # remove x_tick labels and x_axis labels of axis
            axs[0].tick_params(labelbottom=False)
            axs[0].set_xlabel("")
            axs[1].tick_params(labelbottom=False)
            axs[1].set_xlabel("")
            axs[2].tick_params(labelbottom=False)
            axs[2].set_xlabel("")

            # Adjust layout
            plt.tight_layout()

            # Create a unified legend
            # To avoid duplicate labels, create custom handles
            from matplotlib.lines import Line2D

            custom_lines = [
                Line2D(
                    [0],
                    [0],
                    color=styles["kite"]["color"],
                    marker=styles["kite"]["marker"],
                    linestyle=styles["kite"]["linestyle"],
                    label=styles["kite"]["label"],
                ),
                Line2D(
                    [0],
                    [0],
                    color=styles["support"]["color"],
                    marker=styles["support"]["marker"],
                    linestyle=styles["support"]["linestyle"],
                    label=styles["support"]["label"],
                ),
                Line2D(
                    [0],
                    [0],
                    color=styles["total"]["color"],
                    marker=styles["total"]["marker"],
                    linestyle=styles["total"]["linestyle"],
                    label=styles["total"]["label"],
                ),
            ]

            # Place the legend at the top center outside the subplots
            fig.legend(
                handles=custom_lines, loc="upper center", ncol=3
            )  # , fontsize=12)

            # axs[0].legend()
            # plt.legend()

            # Adjust layout to make space for the legend
            plt.subplots_adjust(top=0.93)

            # Define filename
            filename = f"beta_sweep_alpha_{alpha_value:.2f}_kite_support_total"

            # Save the figure
            save_path = Path(results_path) / f"{filename}.pdf"
            save_tex_path = (
                Path(results_path) / f"{filename}.tex"
            )  # Assuming saving_tex is needed

            # Save as PDF
            plt.savefig(save_path, format="pdf")

            print(f"Saved plot for alpha={alpha_value} to {save_path}")

            # Close the figure to free memory
            plt.close(fig)


def main(results_path, project_dir):
    # # Load the data from the CSV file
    # stats_all = loading_data(project_dir)

    # # Plot the data
    plot_speeds = [5, 10, 15, 20, 25]
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

    # alphas_to_be_plotted = [2.35, 4.75, 6.75]
    alphas_to_be_plotted = [6.8]

    ### Other figure settings
    columns = ["C_L", "C_D", "C_S", "C_pitch", "C_roll", "C_yaw"]
    columns = ["C_L", "C_D", "C_S", "C_roll", "C_pitch", "C_yaw"]
    y_labels = ["CL", "CD", "CS", "CMx", "CMy", "CMz"]
    subplot_titles = [
        "Lift coefficient",
        "Drag Coefficient",
        "Side Force coefficient",
        "Pitch moment coefficient",
        "Roll moment coefficient",
        "Yaw moment coefficient",
    ]

    # Define folder_dir
    folder_dir = Path(project_dir) / "processed_data" / "normal_csv"

    # Create stats_all for alpha sweep - Rey plots
    df_all = []
    sideslip = 0
    for file in os.listdir(Path(folder_dir) / f"beta_{sideslip}"):
        # print(f"file: {file}")
        df = pd.read_csv(Path(folder_dir) / f"beta_{sideslip}" / file)
        # print(f"df: {df.columns}")
        df = reduce_df_by_parameter_mean_and_std(df, "aoa_kite")
        df_all.append(df)
    stats_all = pd.concat(df_all)

    # Plot alpha sweep
    plotting_CL_CD_CS_Pitch_Roll_Yaw_vs_alpha_reynolds_sweep(
        results_path,
        stats_all,
        betas_to_be_plotted,
        plot_speeds,
        columns,
        y_labels,
        subplot_titles,
    )

    # Create stats_all for beta sweep - Rey plots
    df_all = []
    alpha = 6.8
    for file in os.listdir(Path(folder_dir) / f"alpha_{alpha}"):
        if "raw" in file:
            continue
        # print(f"file: {file}")
        df = pd.read_csv(Path(folder_dir) / f"alpha_{alpha}" / file)
        # print(f"df: {df.columns}")
        df = reduce_df_by_parameter_mean_and_std(df, "sideslip")
        df_all.append(df)

    stats_all = pd.concat(df_all)

    # Plot beta sweep - Rey
    plotting_CL_CD_CS_Pitch_Roll_Yaw_vs_beta_reynolds_sweep(
        results_path,
        stats_all,
        alphas_to_be_plotted,
        plot_speeds,
        columns,
        y_labels,
        subplot_titles,
    )

    ### plotting Support Structure Loads
    plotting_CL_CD_CS_Pitch_Roll_Yaw_vs_beta_total_kite_support(
        results_path,
        stats_all,
        alphas_to_be_plotted,
        plot_speeds,
        columns,
        y_labels,
        subplot_titles,
    )


if __name__ == "__main__":
    results_path = Path(project_dir) / "results"
    main(results_path, project_dir)
