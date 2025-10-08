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
    apply_angle_wind_tunnel_corrections_to_df,
    alpha_wind_tunnel_correction,
    beta_wind_tunnel_correction,
)
from load_balance_analysis.functions_statistics import (
    calculate_confidence_interval,
    block_bootstrap_confidence_interval,
    hac_newey_west_confidence_interval,
)


# def plot_zigzag(
#     csv_path: str,
#     results_dir: str,
#     confidence_interval: float = 99,
#     max_lag: int = 11,
# ) -> None:

#     # Read the processed data
#     merged_df = pd.read_csv(csv_path)

#     # Rounding the vw values
#     merged_df["vw_actual_rounded"] = np.round(merged_df["vw"], 0)
#     merged_df["Rey_rounded"] = np.round(merged_df["Rey"], -4)

#     # print(f" sideslip_angles: {merged_df['sideslip'].unique()}")

#     print(f'merged_df["vw"].unique(): {merged_df["vw_actual_rounded"].unique()}')
#     print(f'merged df Rey: {merged_df["Rey_rounded"].unique()}')

#     # Defining coefficients
#     coefficients = ["C_L", "C_D", "C_pitch"]
#     yaxis_names = ["CL", "CD", "CS"]

#     data_to_print = []
#     labels_to_print = []
#     rey_list = []
#     for vw in merged_df["vw_actual_rounded"].unique():
#         # finding the data for each vw
#         df_vw = merged_df[merged_df["vw_actual_rounded"] == vw]
#         # calculating reynolds number
#         rey = round(df_vw["Rey"].mean() * 1e-5, 1)
#         if rey == 2.9:
#             rey = 2.8
#         print(f"\nvw: {vw}, rey: {rey}")

#         # separating into zz and no zz
#         data_zz = df_vw[df_vw["Filename"].str.startswith("ZZ")]
#         data_no_zz = df_vw[df_vw["Filename"].str.startswith("normal")]

#         ## rounding the sideslip values in data_zz and data_no_zz
#         # data_zz["sideslip"] = np.round(data_zz["sideslip"], 0)
#         # data_no_zz["sideslip"] = np.round(data_no_zz["sideslip"], 0)

#         # if vw == 15, extracting data for each sideslip
#         if vw == 15:
#             data_zz_sideslip_min10 = data_zz[data_zz["sideslip"] == -10]
#             data_zz_sideslip_0 = data_zz[data_zz["sideslip"] == 0]
#             data_zz_sideslip_10 = data_zz[data_zz["sideslip"] == 10]

#             data_no_zz_sideslip_min10 = data_no_zz[data_no_zz["sideslip"] == -10]
#             data_no_zz_sideslip_0 = data_no_zz[data_no_zz["sideslip"] == 0]
#             data_no_zz_sideslip_10 = data_no_zz[data_no_zz["sideslip"] == 10]
#             # label_list = [
#             #     rf"{rey:.1f} with zigzag ($\beta = -10^\circ$)",
#             #     rf"{rey:.1f} No zigzag ($\beta = -10^\circ$)",
#             #     rf"{rey:.1f} with zigzag ($\beta = 0^\circ$)",
#             #     rf"{rey:.1f} No zigzag ($\beta = 0^\circ$)",
#             #     rf"{rey:.1f} with zigzag ($\beta = 10^\circ$)",
#             #     rf"{rey:.1f} No zigzag ($\beta = 10^\circ$)",
#             # ]
#             # data_list = [
#             #     data_zz_sideslip_min10,
#             #     data_no_zz_sideslip_min10,
#             #     data_zz_sideslip_0,
#             #     data_no_zz_sideslip_0,
#             #     data_zz_sideslip_10,
#             #     data_no_zz_sideslip_10,
#             # ]
#             label_list = [
#                 rf"{rey:.1f} with zigzag ($\beta = -10^\circ$)",
#                 rf"{rey:.1f} No zigzag ($\beta = -10^\circ$)",
#                 rf"{rey:.1f} with zigzag ($\beta = 0^\circ$)",
#                 rf"{rey:.1f} No zigzag ($\beta = 0^\circ$)",
#                 rf"{rey:.1f} with zigzag ($\beta = 10^\circ$)",
#                 rf"{rey:.1f} No zigzag ($\beta = 10^\circ$)",
#             ]
#             data_list = [
#                 data_zz_sideslip_min10,
#                 data_no_zz_sideslip_min10,
#                 data_zz_sideslip_0,
#                 data_no_zz_sideslip_0,
#                 data_zz_sideslip_10,
#                 data_no_zz_sideslip_10,
#             ]
#         else:
#             data_zz_sideslip = data_zz[data_zz["sideslip"] == 0]
#             data_no_zz_sideslip = data_no_zz[data_no_zz["sideslip"] == 0]

#             label_list = [
#                 rf"{rey:.1f} with zigzag",
#                 rf"{rey:.1f} No zigzag",
#             ]
#             data_list = [
#                 data_zz_sideslip,
#                 data_no_zz_sideslip,
#             ]

#         for data, label in zip(data_list, label_list):
#             # print(f"\nlabel: {label}")
#             # skipping the data that has rey = 1.4
#             if "1.4" in label:
#                 continue
#             coefficients = ["C_L", "C_D", "C_S"]

#             ## Standard Deviation
#             data_calculated = [
#                 [data[coeff].mean(), data[coeff].std()] for coeff in coefficients
#             ]
#             ## HAC Newey-West Confidence Interval
#             data_calculated = [
#                 [
#                     data[coeff].mean(),
#                     hac_newey_west_confidence_interval(
#                         data[coeff],
#                         confidence_interval=confidence_interval,
#                         max_lag=max_lag,
#                     ),
#                 ]
#                 for coeff in coefficients
#             ]

#             ## Computing angle differences
#             if "10" in label:
#                 beta_uncorrected = 10
#             elif "-10" in label:
#                 beta_uncorrected = -10
#             else:
#                 beta_uncorrected = 0
#             alpha_corrected = alpha_wind_tunnel_correction(8.95, data["C_L"].mean())
#             beta_corrected = beta_wind_tunnel_correction(
#                 beta_uncorrected, data["C_S"].mean()
#             )
#             print(
#                 f"alpha_corrected: {alpha_corrected}, beta_corrected: {beta_corrected}"
#             )

#             ## Appending
#             rey_list.append(rey)
#             data_to_print.append(data_calculated)
#             labels_to_print.append(label)

#     # print(f"data_to_print: {data_to_print}")
#     # print(f"labels_to_print: {labels_to_print}")
#     # print(f"rey_list: {rey_list}")

#     create_grouped_plot(
#         rey_list,
#         data_to_print,
#         labels_to_print,
#         y_axis_labels,
#         yaxis_names,
#         results_dir,
#     )


# def create_grouped_plot(
#     rey_list, data_to_print, labels_to_print, y_axis_labels, yaxis_names, results_dir
# ):
#     # this will be a 1x3 plot, so we do 15x5
#     fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
#     # )  # Increased figure height
#     axs = axes.flatten()

#     # Group data by Reynolds number
#     reynolds_groups = {2.5: [], 3.8: [], 5.0: []}

#     # Add shaded regions first so they appear behind the data points
#     shaded_regions = [0, 2]  # Indices
#     for ax in axs:
#         for region_idx in shaded_regions:
#             ax.axvspan(region_idx - 0.5, region_idx + 0.5, color="gray", alpha=0.15)

#     for rey, data, label in zip(rey_list, data_to_print, labels_to_print):
#         # Determine which Reynolds number group this belongs to
#         for key in reynolds_groups.keys():
#             if abs(rey - key) < 0.6:  # Use approximate matching
#                 reynolds_groups[key].append((data, label))
#                 break

#     # Set up x-axis
#     group_names = list(reynolds_groups.keys())
#     n_groups = len(group_names)

#     for ax_idx, ax in enumerate(axs):
#         for group_idx, (rey_num, group_data) in enumerate(reynolds_groups.items()):
#             if len(group_data) == 2:
#                 x_positions = np.linspace(
#                     group_idx - 0.05, group_idx + 0.05, len(group_data)
#                 )
#                 color_list = ["red", "blue"]
#                 marker_list = ["o", "o"]
#             elif len(group_data) == 6:
#                 x_positions = np.linspace(
#                     group_idx - 0.25, group_idx + 0.25, len(group_data)
#                 )
#                 color_list = [
#                     "red",
#                     "blue",
#                     "red",
#                     "blue",
#                     "red",
#                     "blue",
#                 ]
#                 marker_list = ["v", "v", "o", "o", "^", "^"]

#             for x_pos, (data, label), color, marker in zip(
#                 x_positions, group_data, color_list, marker_list
#             ):
#                 ax.errorbar(
#                     x_pos,
#                     data[ax_idx][0],
#                     yerr=data[ax_idx][1],
#                     fmt=marker,
#                     color=color,
#                     capsize=5,
#                 )

#         # Set x-axis
#         # print(f"n_groups: {n_groups}")
#         # print(f"group_names: {group_names}")
#         ax.set_xticks(range(n_groups))
#         ax.set_xticklabels(group_names)
#         ax.set_xlabel(r"Re $\times 10^5$ [-]")
#         ax.set_ylabel(y_axis_labels[yaxis_names[ax_idx]])
#         # Set only horizontal gridlines
#         ax.grid(True, axis="y")
#         ax.grid(False, axis="x")

#         for ax in axs:
#             # Get current x-axis limits
#             xlim = ax.get_xlim()

#             # Extend x-axis limits slightly on both sides
#             padding = 0.19  # Adjust this value to control the gray area size
#             ax.set_xlim(xlim[0] + padding, xlim[1] - padding)

#     axs[0].set_ylim(0.65, 0.90)
#     # Create legend elements
#     legend_elements = [
#         plt.Line2D(
#             [0],
#             [0],
#             marker="o",
#             color="red",
#             label=r"With zigzag $\beta = 0^\circ$ ",
#             markersize=8,
#             linestyle="None",
#         ),
#         plt.Line2D(
#             [0],
#             [0],
#             marker="o",
#             color="blue",
#             label=r"No zigzag $\beta = 0^\circ$",
#             markersize=8,
#             linestyle="None",
#         ),
#         plt.Line2D(
#             [0],
#             [0],
#             marker="v",
#             color="red",
#             label=r"With zigzag $\beta = -10^\circ$",
#             markersize=8,
#             linestyle="None",
#         ),
#         plt.Line2D(
#             [0],
#             [0],
#             marker="v",
#             color="blue",
#             label=r"No zigzag $\beta = -10^\circ$",
#             markersize=8,
#             linestyle="None",
#         ),
#         plt.Line2D(
#             [0],
#             [0],
#             marker="^",
#             color="red",
#             label=r"With zigzag $\beta = 10^\circ$",
#             markersize=8,
#             linestyle="None",
#         ),
#         plt.Line2D(
#             [0],
#             [0],
#             marker="^",
#             color="blue",
#             label=r"No zigzag $\beta = 10^\circ$",
#             markersize=8,
#             linestyle="None",
#         ),
#     ]

#     # Add centered legend below the plots
#     fig.legend(
#         handles=legend_elements,
#         loc="lower center",  # Changed to lower center
#         bbox_to_anchor=(0.5, 0.0),  # Adjust y value negative to push further down
#         ncol=3,
#         frameon=True,
#     )

#     plt.tight_layout()

#     # Increase bottom margin more significantly
#     plt.subplots_adjust(bottom=0.28)  # Increased from 0.24

#     # Save the figure
#     saving_pdf_and_pdf_tex(results_dir, "zz_re_sweep_alpha_8_beta_0")


# def main(results_dir: Path, project_dir: Path) -> None:

#     # # Increase font size for readability
#     # fontsize = 18

#     save_path_lvm_data_processed = (
#         Path(project_dir)
#         / "processed_data"
#         / "zigzag_csv"
#         / "alpha_8.9"
#         / "lvm_data_processed.csv"
#     )
#     plot_zigzag(
#         save_path_lvm_data_processed,
#         results_dir,
#         confidence_interval=99,
#         max_lag=11,
#     )


# if __name__ == "__main__":

#     results_dir = Path(project_dir) / "results"
#     main(results_dir, project_dir)


def plot_zigzag(
    csv_path: str,
    results_dir: str,
    confidence_interval: float = 99,
    max_lag: int = 11,
) -> None:

    # Read the processed data
    merged_df = pd.read_csv(csv_path)

    # Rounding the vw values
    merged_df["vw_actual_rounded"] = np.round(merged_df["vw"], 0)
    merged_df["Rey_rounded"] = np.round(merged_df["Rey"], -4)

    # print(f" sideslip_angles: {merged_df['sideslip'].unique()}")

    print(f'merged_df["vw"].unique(): {merged_df["vw_actual_rounded"].unique()}')
    print(f'merged df Rey: {merged_df["Rey_rounded"].unique()}')

    # Defining coefficients
    coefficients = ["C_L", "C_D", "C_pitch"]
    yaxis_names = ["CL", "CD", "CS"]

    data_to_print = []
    labels_to_print = []
    rey_list = []
    for vw in merged_df["vw_actual_rounded"].unique():
        # finding the data for each vw
        df_vw = merged_df[merged_df["vw_actual_rounded"] == vw]
        # calculating reynolds number
        rey = round(df_vw["Rey"].mean() * 1e-5, 1)
        if rey == 2.9:
            rey = 2.8
        print(f"\nvw: {vw}, rey: {rey}")

        # separating into zz and no zz
        data_zz = df_vw[df_vw["Filename"].str.startswith("ZZ")]
        data_no_zz = df_vw[df_vw["Filename"].str.startswith("normal")]

        ## rounding the sideslip values in data_zz and data_no_zz
        # data_zz["sideslip"] = np.round(data_zz["sideslip"], 0)
        # data_no_zz["sideslip"] = np.round(data_no_zz["sideslip"], 0)

        # if vw == 15, extracting data for each sideslip
        if vw == 15:
            data_zz_sideslip_min10 = data_zz[data_zz["sideslip"] == -10]
            data_zz_sideslip_0 = data_zz[data_zz["sideslip"] == 0]
            data_zz_sideslip_10 = data_zz[data_zz["sideslip"] == 10]

            data_no_zz_sideslip_min10 = data_no_zz[data_no_zz["sideslip"] == -10]
            data_no_zz_sideslip_0 = data_no_zz[data_no_zz["sideslip"] == 0]
            data_no_zz_sideslip_10 = data_no_zz[data_no_zz["sideslip"] == 10]

            # Changed order here: "No zigzag" first, then "with zigzag"
            label_list = [
                rf"{rey:.1f} No zigzag ($\beta = -10^\circ$)",
                rf"{rey:.1f} with zigzag ($\beta = -10^\circ$)",
                rf"{rey:.1f} No zigzag ($\beta = 0^\circ$)",
                rf"{rey:.1f} with zigzag ($\beta = 0^\circ$)",
                rf"{rey:.1f} No zigzag ($\beta = 10^\circ$)",
                rf"{rey:.1f} with zigzag ($\beta = 10^\circ$)",
            ]
            data_list = [
                data_no_zz_sideslip_min10,
                data_zz_sideslip_min10,
                data_no_zz_sideslip_0,
                data_zz_sideslip_0,
                data_no_zz_sideslip_10,
                data_zz_sideslip_10,
            ]
        else:
            data_zz_sideslip = data_zz[data_zz["sideslip"] == 0]
            data_no_zz_sideslip = data_no_zz[data_no_zz["sideslip"] == 0]

            # Changed order here: "No zigzag" first, then "with zigzag"
            label_list = [
                rf"{rey:.1f} No zigzag",
                rf"{rey:.1f} with zigzag",
            ]
            data_list = [
                data_no_zz_sideslip,
                data_zz_sideslip,
            ]

        for data, label in zip(data_list, label_list):
            # print(f"\nlabel: {label}")
            # skipping the data that has rey = 1.4
            if "1.4" in label:
                continue
            coefficients = ["C_L", "C_D", "C_S"]

            ## Standard Deviation
            data_calculated = [
                [data[coeff].mean(), data[coeff].std()] for coeff in coefficients
            ]
            ## HAC Newey-West Confidence Interval
            data_calculated = [
                [
                    data[coeff].mean(),
                    hac_newey_west_confidence_interval(
                        data[coeff],
                        confidence_interval=confidence_interval,
                        max_lag=max_lag,
                    ),
                ]
                for coeff in coefficients
            ]

            ## Computing angle differences
            if "10" in label:
                beta_uncorrected = 10
            elif "-10" in label:
                beta_uncorrected = -10
            else:
                beta_uncorrected = 0
            alpha_corrected = alpha_wind_tunnel_correction(8.95, data["C_L"].mean())
            beta_corrected = beta_wind_tunnel_correction(
                beta_uncorrected, data["C_S"].mean()
            )
            print(
                f"alpha_corrected: {alpha_corrected}, beta_corrected: {beta_corrected}"
            )

            ## Appending
            rey_list.append(rey)
            data_to_print.append(data_calculated)
            labels_to_print.append(label)

    # print(f"data_to_print: {data_to_print}")
    # print(f"labels_to_print: {labels_to_print}")
    # print(f"rey_list: {rey_list}")

    create_grouped_plot(
        rey_list,
        data_to_print,
        labels_to_print,
        y_axis_labels,
        yaxis_names,
        results_dir,
    )


def create_grouped_plot(
    rey_list, data_to_print, labels_to_print, y_axis_labels, yaxis_names, results_dir
):
    # this will be a 1x3 plot, so we do 15x5
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    # )  # Increased figure height
    axs = axes.flatten()

    # Group data by Reynolds number
    reynolds_groups = {2.5: [], 3.8: [], 5.0: []}

    for rey, data, label in zip(rey_list, data_to_print, labels_to_print):
        # Determine which Reynolds number group this belongs to
        for key in reynolds_groups.keys():
            if abs(rey - key) < 0.6:  # Use approximate matching
                reynolds_groups[key].append((data, label))
                break

    # Set up x-axis
    group_names = list(reynolds_groups.keys())
    n_groups = len(group_names)

    for ax_idx, ax in enumerate(axs):
        for group_idx, (rey_num, group_data) in enumerate(reynolds_groups.items()):
            if len(group_data) == 2:
                x_positions = np.linspace(
                    group_idx - 0.05, group_idx + 0.05, len(group_data)
                )
                # Changed color order to match "No zigzag" first, "with zigzag" second
                color_list = ["blue", "red"]
                color_list = ["black", "red"]
                marker_list = ["o", "o"]
            elif len(group_data) == 6:
                x_positions = np.linspace(
                    group_idx - 0.25, group_idx + 0.25, len(group_data)
                )
                # Changed color order to match "No zigzag" first, "with zigzag" second
                color_list = [
                    "blue",
                    "red",
                    "blue",
                    "red",
                    "blue",
                    "red",
                ]
                color_list = [
                    "black",
                    "red",
                    "black",
                    "red",
                    "black",
                    "red",
                ]
                marker_list = ["v", "v", "o", "o", "^", "^"]

            for x_pos, (data, label), color, marker in zip(
                x_positions, group_data, color_list, marker_list
            ):
                ax.errorbar(
                    x_pos,
                    data[ax_idx][0],
                    yerr=data[ax_idx][1],
                    fmt=marker,
                    color=color,
                    capsize=5,
                )

        # Set x-axis
        # print(f"n_groups: {n_groups}")
        # print(f"group_names: {group_names}")
        ax.set_xticks(range(n_groups))
        ax.set_xticklabels(group_names)
        ax.set_xlabel(r"Re $\times 10^5$ (-)")
        ax.set_ylabel(y_axis_labels[yaxis_names[ax_idx]])
        # Set only horizontal gridlines
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")

    # Add vertical separator lines after all data is plotted for all subplots
    for ax in axs:
        # Add vertical separator lines between groups (0.5 is between groups 0 and 1, 1.5 is between groups 1 and 2)
        separator_positions = [0.5, 1.5]  # Lines between groups
        for sep_pos in separator_positions:
            ax.axvline(x=sep_pos, color="gray", linestyle="-", linewidth=1, alpha=0.5)

        # Set tighter x-axis limits to make bins appear equally spaced
        ax.set_xlim(
            -0.4, 2.4
        )  # From -0.4 to 2.4 to give equal spacing around 3 groups (0, 1, 2)

    axs[0].set_ylim(0.65, 0.90)
    # Create legend elements - Changed order to "No zigzag" first, "With zigzag" second
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color=color_list[0],
            label=r"No zigzag $\beta = 0^\circ$",
            markersize=8,
            linestyle="None",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color=color_list[1],
            label=r"With zigzag $\beta = 0^\circ$ ",
            markersize=8,
            linestyle="None",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="v",
            color=color_list[0],
            label=r"No zigzag $\beta = -10^\circ$",
            markersize=8,
            linestyle="None",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="v",
            color=color_list[1],
            label=r"With zigzag $\beta = -10^\circ$",
            markersize=8,
            linestyle="None",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="^",
            color=color_list[0],
            label=r"No zigzag $\beta = 10^\circ$",
            markersize=8,
            linestyle="None",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="^",
            color=color_list[1],
            label=r"With zigzag $\beta = 10^\circ$",
            markersize=8,
            linestyle="None",
        ),
    ]

    # Remove x-tick marks but keep labels
    for ax in axs:
        ax.tick_params(axis="x", length=0)

    # Add centered legend below the plots
    fig.legend(
        handles=legend_elements,
        loc="lower center",  # Changed to lower center
        bbox_to_anchor=(0.5, 0.0),  # Adjust y value negative to push further down
        ncol=3,
        frameon=False,
    )

    plt.tight_layout()

    # Increase bottom margin more significantly
    plt.subplots_adjust(bottom=0.28)  # Increased from 0.24

    # Save the figure
    saving_pdf_and_pdf_tex(results_dir, "zz_re_sweep_alpha_8_beta_0")


def main(results_dir: Path, project_dir: Path) -> None:

    # # Increase font size for readability
    # fontsize = 18

    save_path_lvm_data_processed = (
        Path(project_dir)
        / "processed_data"
        / "zigzag_csv"
        / "alpha_9.9"
        / "lvm_data_processed.csv"
    )
    plot_zigzag(
        save_path_lvm_data_processed,
        results_dir,
        confidence_interval=99,
        max_lag=11,
    )


if __name__ == "__main__":

    results_dir = Path(project_dir) / "results"
    main(results_dir, project_dir)
