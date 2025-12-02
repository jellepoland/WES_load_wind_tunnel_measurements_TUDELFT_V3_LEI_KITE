import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path
from load_balance_analysis.functions_utils import (
    project_dir,
    saving_pdf_and_pdf_tex,
    x_axis_labels,
    y_axis_labels,
    reduce_df_by_parameter_mean_and_std,
    apply_angle_wind_tunnel_corrections_to_df,
)
from load_balance_analysis.functions_statistics import (
    hac_newey_west_confidence_interval,
)
from matplotlib.patches import Patch
from plot_styling import plot_on_ax


def plot_single_row(
    results_dir,
    data_frames,
    labels,
    colors,
    linestyles,
    markers,
    file_name,
    confidence_interval,
    axs_titles,
    legend_location_index=0,
    legend_location="lower left",
    variables_to_plot=["CL", "CD", "CS"],  # <--- NEW: which columns to plot
    show_ci=True,  # <--- NEW: toggle confidence intervals
    xlim=None,
    ylim=None,
):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # Three subplots in one row

    for i, (data_frame, label, color, linestyle, marker) in enumerate(
        zip(data_frames, labels, colors, linestyles, markers)
    ):
        if data_frame is None:
            continue

        # Some runs might have a ratio for side-force 'CS'
        # e.g. 3D geometry difference vs. 2D reference area
        if "CFD" in label:
            ratio_projected_area_to_side_area = 1.0
        else:
            ratio_projected_area_to_side_area = 1.0

        # Marker size only for the first dataset
        markersize_i = 8 if i == 0 else None

        # Plot each chosen variable on its corresponding subplot
        for j, var in enumerate(variables_to_plot):
            # X-values are always "beta" in your setup
            x_data = data_frame["beta"]
            # Y-values may need special scaling if var == "CS"
            if var == "CS":
                y_data = data_frame[var] / ratio_projected_area_to_side_area
            else:
                y_data = data_frame[var]

            # Plot main data
            plot_on_ax(
                axs[j],
                x_data,
                y_data,
                label=label,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=markersize_i,
            )

        # If this is the last dataset (i == 2) AND we want confidence intervals:
        if "WT" in label and show_ci:
            # For each variable, fill between upper/lower bounds if *_ci is in DataFrame
            for j, var in enumerate(variables_to_plot):
                # Build main Y data
                if var == "CS":
                    main_y = data_frame[var] / ratio_projected_area_to_side_area
                    ci_y = data_frame[var + "_ci"] / ratio_projected_area_to_side_area
                else:
                    main_y = data_frame[var]
                    ci_y = data_frame[var + "_ci"] if (var + "_ci") in data_frame else 0

                lower_bound = main_y - ci_y
                upper_bound = main_y + ci_y

                # Fill the CI region on positive beta
                axs[j].fill_between(
                    data_frame["beta"],
                    lower_bound,
                    upper_bound,
                    color=color,
                    alpha=0.15,
                    label=f"WT CI of {confidence_interval}\%",
                )

            ### Also plot the negative-beta mirror curves:
            # You show negative-beta versions of CL,CD,CS with dashed lines
            for j, var in enumerate(variables_to_plot):
                if var == "CS" or var == "CMx" or var == "CMz":
                    main_y = data_frame[var] / ratio_projected_area_to_side_area
                    ci_y = data_frame[var + "_ci"] / ratio_projected_area_to_side_area
                    # Negative side: sign flips for CS
                    main_y_neg = -main_y
                else:
                    main_y = data_frame[var]
                    ci_y = data_frame[var + "_ci"] if (var + "_ci") in data_frame else 0
                    # Negative side: sign does NOT flip for CL, CD, etc.
                    main_y_neg = main_y

                axs[j].plot(
                    -data_frame["beta"],
                    main_y_neg,
                    color=color,
                    linestyle=linestyle.replace("-", "--"),
                    marker=marker,
                    markersize=markersize_i,
                    label=label + r" , -$\beta$",
                )

                # If also filling CI for negative beta:
                lower_bound = main_y - ci_y
                upper_bound = main_y + ci_y
                # Sign flip only if var=="CS"
                if var == "CS" or var == "CMx" or var == "CMz":
                    lower_bound_neg = -lower_bound
                    upper_bound_neg = -upper_bound
                else:
                    lower_bound_neg = lower_bound
                    upper_bound_neg = upper_bound

                axs[j].fill_between(
                    -data_frame["beta"],
                    lower_bound_neg,
                    upper_bound_neg,
                    color="white",
                    facecolor=color,
                    alpha=0.3,
                    label=rf"WT CI of {confidence_interval}\% , -$\beta$",
                    hatch="||",
                )

    # Final axis formatting
    # X-limits, labels, titles, etc.
    for j in range(3):
        axs[j].set_xlim(0, 20)
        axs[j].set_xlabel(r"$\beta$ ($^{\circ}$)")
        # Optional: if you have a dictionary for custom axis titles:
        # axs[j].set_title(axs_titles[j])

    # For demonstration, label Y-axes based on variables_to_plot
    # (You could also pass in a separate param of y-axis labels if you prefer.)
    axs[0].set_ylabel(y_axis_labels[variables_to_plot[0]])
    axs[1].set_ylabel(y_axis_labels[variables_to_plot[1]])
    axs[2].set_ylabel(y_axis_labels[variables_to_plot[2]])

    # Example special Y-limits if "CFD" in the first label
    if len(labels) > 0 and "CFD" in labels[0]:
        axs[2].set_ylim(-0.05, 0.3)
    elif "CS" in variables_to_plot:
        axs[2].set_ylim(-0.05, 0.5)

    if xlim is not None:
        axs[0].set_xlim(xlim[0])
        axs[1].set_xlim(xlim[1])
        axs[2].set_xlim(xlim[2])
    if ylim is not None:
        axs[0].set_ylim(ylim[0])
        axs[1].set_ylim(ylim[1])
        axs[2].set_ylim(ylim[2])

    # Adjust layout for subplots
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.28)

    # Create custom legend elements for final figure
    legend_elements = []
    last_i = len(data_frames) - 1  # Default last index for confidence intervals
    for i in range(last_i):
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker=markers[i],
                color=colors[i],
                label=labels[i],
                linestyle=linestyles[i],
            )
        )
    legend_elements_WT = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="red",
            label=r"WT Re = $5 \times 10^5$",
            linestyle="-",
        ),
        Patch(facecolor="red", edgecolor="none", alpha=0.15, label=r"WT CI of 99\%"),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="red",
            linestyle="--",
            label=r"WT Re = $5 \times 10^5$ $(-\beta)$",
        ),
        Patch(
            facecolor="red",
            edgecolor="white",
            alpha=0.3,
            hatch="||",
            label=r"WT CI of 99\% $(-\beta)$",
        ),
    ]
    legend_elements.extend(legend_elements_WT)

    # Final legend at bottom
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=False,
    )

    # Finally save
    saving_pdf_and_pdf_tex(results_dir, file_name)


def plot_double_row(
    results_dir,
    high_data_frames,
    high_labels,
    high_colors,
    high_linestyles,
    high_markers,
    low_data_frames,
    low_labels,
    low_colors,
    low_linestyles,
    low_markers,
    file_name,
    confidence_interval,
    axs_titles_high,
    axs_titles_low,
    variables_to_plot=["CL", "CD", "CS"],
    show_ci=True,
    xlim=None,
    ylim=None,
):
    # Create a figure with 2 rows and 3 columns
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))

    # ---------- Top Row: High Alpha Data ----------
    for i, (data_frame, label, color, linestyle, marker) in enumerate(
        zip(high_data_frames, high_labels, high_colors, high_linestyles, high_markers)
    ):
        if data_frame is None:
            continue

        # For some cases, side-force may require a scaling ratio
        ratio_projected_area_to_side_area = 1.0 if "CFD" in label else 1.0
        markersize_i = 8 if i == 0 else None

        # Plot each variable on its corresponding subplot (columns)
        for j, var in enumerate(variables_to_plot):
            x_data = data_frame["beta"]
            y_data = (
                data_frame[var] / ratio_projected_area_to_side_area
                if var == "CS"
                else data_frame[var]
            )
            plot_on_ax(
                axs[0, j],
                x_data,
                y_data,
                label=label,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=markersize_i,
                is_with_x_label=False,
                is_with_x_ticks=False,
                y_label=(
                    r"High $\alpha = 12.5^{\circ}$" + f"\n \n" + y_axis_labels[var]
                    if j == 0
                    else y_axis_labels[var]
                ),
            )

        # Optionally plot confidence intervals for "WT" data
        if "WT" in label and show_ci:
            # Fill CI on positive beta
            for j, var in enumerate(variables_to_plot):
                if var == "CS":
                    main_y = data_frame[var] / ratio_projected_area_to_side_area
                    ci_y = data_frame[var + "_ci"] / ratio_projected_area_to_side_area
                else:
                    main_y = data_frame[var]
                    ci_y = data_frame[var + "_ci"] if (var + "_ci") in data_frame else 0
                lower_bound = main_y - ci_y
                upper_bound = main_y + ci_y
                axs[0, j].fill_between(
                    data_frame["beta"],
                    lower_bound,
                    upper_bound,
                    color=color,
                    alpha=0.15,
                    label=f"WT CI of {confidence_interval}\%",
                )
            # Also plot negative-beta mirror curves
            for j, var in enumerate(variables_to_plot):
                if var in ["CS", "CMx", "CMz"]:
                    main_y = data_frame[var] / ratio_projected_area_to_side_area
                    ci_y = data_frame[var + "_ci"] / ratio_projected_area_to_side_area
                    main_y_neg = -main_y
                else:
                    main_y = data_frame[var]
                    ci_y = data_frame[var + "_ci"] if (var + "_ci") in data_frame else 0
                    main_y_neg = main_y
                axs[0, j].plot(
                    -data_frame["beta"],
                    main_y_neg,
                    color=color,
                    linestyle=linestyle.replace("-", "--"),
                    marker=marker,
                    markersize=markersize_i,
                    label=label + r" , -$\beta$",
                )
                lower_bound = main_y - ci_y
                upper_bound = main_y + ci_y
                if var in ["CS", "CMx", "CMz"]:
                    lower_bound_neg = -lower_bound
                    upper_bound_neg = -upper_bound
                else:
                    lower_bound_neg = lower_bound
                    upper_bound_neg = upper_bound
                axs[0, j].fill_between(
                    -data_frame["beta"],
                    lower_bound_neg,
                    upper_bound_neg,
                    color="white",
                    facecolor=color,
                    alpha=0.3,
                    label=rf"WT CI of {confidence_interval}\% , -$\beta$",
                    hatch="||",
                )

    # ---------- Bottom Row: Low Alpha Data ----------
    for i, (data_frame, label, color, linestyle, marker) in enumerate(
        zip(low_data_frames, low_labels, low_colors, low_linestyles, low_markers)
    ):
        if data_frame is None:
            continue

        ratio_projected_area_to_side_area = 1.0 if "CFD" in label else 1.0
        markersize_i = 8 if i == 0 else None

        for j, var in enumerate(variables_to_plot):
            x_data = data_frame["beta"]
            y_data = (
                data_frame[var] / ratio_projected_area_to_side_area
                if var == "CS"
                else data_frame[var]
            )
            # FIX: Use axs[1, j] for second row
            plot_on_ax(
                axs[1, j],
                x_data,
                y_data,
                label=None,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=markersize_i,
                x_label=r"$\beta (^{\circ})$",
                y_label=(
                    r"Low $\alpha = 7.4^{\circ}$" + f"\n \n" + y_axis_labels[var]
                    if j == 0
                    else y_axis_labels[var]
                ),
            )

        # Optionally plot confidence intervals for "WT" data
        if "WT" in label and show_ci:
            # Fill CI on positive beta
            for j, var in enumerate(variables_to_plot):
                if var == "CS":
                    main_y = data_frame[var] / ratio_projected_area_to_side_area
                    ci_y = data_frame[var + "_ci"] / ratio_projected_area_to_side_area
                else:
                    main_y = data_frame[var]
                    ci_y = data_frame[var + "_ci"] if (var + "_ci") in data_frame else 0
                lower_bound = main_y - ci_y
                upper_bound = main_y + ci_y
                axs[1, j].fill_between(
                    data_frame["beta"],
                    lower_bound,
                    upper_bound,
                    color=color,
                    alpha=0.15,
                    label=f"WT CI of {confidence_interval}\%",
                )
            # Also plot negative-beta mirror curves
            for j, var in enumerate(variables_to_plot):
                if var in ["CS", "CMx", "CMz"]:
                    main_y = data_frame[var] / ratio_projected_area_to_side_area
                    ci_y = data_frame[var + "_ci"] / ratio_projected_area_to_side_area
                    main_y_neg = -main_y
                else:
                    main_y = data_frame[var]
                    ci_y = data_frame[var + "_ci"] if (var + "_ci") in data_frame else 0
                    main_y_neg = main_y
                axs[1, j].plot(
                    -data_frame["beta"],
                    main_y_neg,
                    color=color,
                    linestyle=linestyle.replace("-", "--"),
                    marker=marker,
                    markersize=markersize_i,
                    label=label + r" , -$\beta$",
                )
                lower_bound = main_y - ci_y
                upper_bound = main_y + ci_y
                if var in ["CS", "CMx", "CMz"]:
                    lower_bound_neg = -lower_bound
                    upper_bound_neg = -upper_bound
                else:
                    lower_bound_neg = lower_bound
                    upper_bound_neg = upper_bound
                axs[1, j].fill_between(
                    -data_frame["beta"],
                    lower_bound_neg,
                    upper_bound_neg,
                    color="white",
                    facecolor=color,
                    alpha=0.3,
                    label=rf"WT CI of {confidence_interval}\% , -$\beta$",
                    hatch="||",
                )

    # ---------- Final Axis Formatting ----------
    for row in range(2):
        for j in range(3):
            axs[row, j].set_xlim(0, 20)
            # axs[row, j].set_xlabel(r"$\beta$ [^{\circ}]")
            # Optionally, set y-axis labels on left-most column only
            # if j == 0:
            # axs[row, j].set_ylabel(y_axis_labels[variables_to_plot[0]])

    # Set column titles for each row using the provided axis titles
    # for j in range(3):
    # axs[0, j].set_title(axs_titles_high[j])
    # axs[1, j].set_title(axs_titles_low[j])

    # Apply global xlim/ylim if provided (assumed as a list of three values for columns)
    if "CL" in variables_to_plot:
        axs[0, 1].set_ylim(0.05, 0.25)
        axs[0, 2].set_ylim(-0.05, 0.2)  # -0.20, 0.05)
        # axs[1, 0].set_ylim(0.45, 0.8)
        axs[1, 2].set_ylim(-0.05, 0.25)  # -0.25, 0.05)
    elif "CMx" in variables_to_plot:
        axs[0, 0].set_ylim(-0.3, 0.1)
        # axs[0, 1].set_ylim(-0.3, 0.05)
        # axs[0, 2].set_ylim(-0.3, 0.05)

        axs[1, 0].set_ylim(-0.3, 0.1)
        axs[1, 1].set_ylim(-0.1, 0.2)
        axs[1, 2].set_ylim(-0.1, 0.2)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)

    # ---------- Create Combined Legend ----------
    legend_elements = []
    # Legends for high alpha datasets
    for i in range(len(high_data_frames)):
        # print(f"high_labels[i]: {high_labels[i]}")
        # if high_labels[i] == "CFD Re = $10\times10^5$ (Struts)":
        #     label = high_labels[i] + rf" $\alpha = 12$^{\circ}"
        # else:
        #     print(f"VSM or WT in high_labels[i]: {high_labels[i]}")
        #     label = high_labels[i] + rf" $\alpha = 6.5$, $11.6$^{\circ}"

        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker=high_markers[i],
                color=high_colors[i],
                label=high_labels[i],
                linestyle=high_linestyles[i],
            )
        )
    # # Legends for low alpha datasets
    # for i in range(len(low_data_frames)):
    #     legend_elements.append(
    #         plt.Line2D(
    #             [0],
    #             [0],
    #             marker=low_markers[i],
    #             color=low_colors[i],
    #             label=low_labels[i],
    #             linestyle=low_linestyles[i],
    #         )
    #     )
    # Optionally, add legend items for WT confidence intervals
    legend_elements_WT = [
        # plt.Line2D(
        #     [0],
        #     [0],
        #     marker="o",
        #     color="red",
        #     label=r"WT Re = $5 \times 10^5$",
        #     linestyle="-",
        # ),
        Patch(facecolor="red", edgecolor="none", alpha=0.15, label="WT CI of 99\%"),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="red",
            linestyle="--",
            label=r"WT Re = $5 \times 10^5$ $(-\beta)$",
        ),
        Patch(
            facecolor="red",
            edgecolor="white",
            alpha=0.3,
            hatch="||",
            label=r"WT CI of 99\% $(-\beta)$",
        ),
    ]
    legend_elements.extend(legend_elements_WT)

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=False,
    )

    # ---------- Save the Figure ----------
    saving_pdf_and_pdf_tex(results_dir, file_name)


def plotting_polars_beta(
    project_dir: str,
    results_dir: str,
    confidence_interval: float,
):
    polar_dir = Path(project_dir) / "processed_data" / "polar_data"
    # Load VSM data
    # data_VSM_beta_re_56e4_alpha_675_breukels = pd.read_csv(
    #     Path(polar_dir) / f"VSM_results_beta_sweep_Rey_5.0_alpha_740_breukels.csv"
    # )
    # data_VSM_beta_re_56e4_alpha_675_breukels_stall = pd.read_csv(
    #     Path(polar_dir) / f"VSM_results_beta_sweep_Rey_5.0_alpha_740_breukels_stall.csv"
    # )
    data_VSM_beta_re_56e4_alpha_675_corrected = pd.read_csv(
        Path(polar_dir) / f"VSM_results_beta_sweep_Rey_5.0_alpha_740_corrected.csv"
    )
    # data_VSM_beta_re_56e4_alpha_675_corrected_stall = pd.read_csv(
    #     Path(polar_dir)
    #     / f"VSM_results_beta_sweep_Rey_5.0_alpha_740_corrected_stall.csv"
    # )
    # data_VSM_beta_re_56e4_alpha_1195_breukels = pd.read_csv(
    #     Path(polar_dir) / f"VSM_results_beta_sweep_Rey_5.0_alpha_1250_breukels.csv"
    # )
    # data_VSM_beta_re_56e4_alpha_1195_breukels_stall = pd.read_csv(
    #     Path(polar_dir)
    #     / f"VSM_results_beta_sweep_Rey_5.0_alpha_1195_breukels_stall.csv"
    # )
    data_VSM_beta_re_56e4_alpha_1195_corrected = pd.read_csv(
        Path(polar_dir) / f"VSM_results_beta_sweep_Rey_5.1_alpha_1250_corrected.csv"
    )
    # data_VSM_beta_re_56e4_alpha_1195_corrected_stall = pd.read_csv(
    #     Path(polar_dir)
    #     / f"VSM_results_beta_sweep_Rey_5.0_alpha_1250_corrected_stall.csv"
    # )

    # Load Lebesque data
    path_to_csv_lebesque_re_100e4_alpha_1195 = (
        Path(project_dir)
        / "data"
        / "CFD_polar_data"
        / "CFD_RANS_Rey_10e5_Poland2025_beta_sweep_alpha_13_02.csv"
    )
    data_lebesque_re_100e4_alpha_1195 = pd.read_csv(
        path_to_csv_lebesque_re_100e4_alpha_1195
    )

    # Load Wind Tunnel data
    data_WT_beta_re_56e4_alpha_7_7 = pd.read_csv(
        Path(project_dir)
        / "processed_data"
        / "polar_data"
        / "V3_CL_CD_CS_beta_sweep_alpha_7.7_WindTunnel_Poland_2025_Rey_560e4.csv"
    )
    data_WT_beta_re_56e4_alpha_12_9 = pd.read_csv(
        Path(project_dir)
        / "processed_data"
        / "polar_data"
        / "V3_CL_CD_CS_beta_sweep_alpha_12.9_WindTunnel_Poland_2025_Rey_560e4.csv"
    )
    # ## correcting the beta
    # data_WT_beta_re_56e4_alpha_12_9["beta"] = -data_WT_beta_re_56e4_alpha_12_9["beta"]
    # data_WT_beta_re_56e4_alpha_7_7["beta"] = -data_WT_beta_re_56e4_alpha_7_7["beta"]

    ### 1. literature_polars_beta_high_alpha
    plot_single_row(
        results_dir,
        data_frames=[
            data_lebesque_re_100e4_alpha_1195,
            data_VSM_beta_re_56e4_alpha_1195_corrected,
            data_WT_beta_re_56e4_alpha_12_9,
        ],
        labels=[
            rf"CFD Re = $10\times10^5$ (Struts)",
            rf"VSM Re = $5\times10^5$",
            rf"WT Re = $5\times10^5$",
        ],
        colors=["black", "blue", "red"],
        linestyles=["-", "-", "-"],
        markers=["*", ".", "o"],
        file_name=f"literature_polars_beta_high_alpha",
        confidence_interval=confidence_interval,
        axs_titles=["CL (High Alpha)", "CD (High Alpha)", "CS (High Alpha)"],
        legend_location_index=0,
        legend_location="lower left",
    )
    ### 2. literature_polars_beta_low_alpha
    plot_single_row(
        results_dir,
        data_frames=[
            data_VSM_beta_re_56e4_alpha_675_corrected,
            data_WT_beta_re_56e4_alpha_7_7,
        ],
        labels=[
            # rf"VSM Re = $5\times10^5$",
            rf"VSM Re = $5\times10^5$",
            rf"WT Re = $5\times10^5$",
        ],
        colors=["blue", "red"],
        linestyles=["-", "-"],
        markers=[".", "o"],
        confidence_interval=confidence_interval,
        file_name=f"literature_polars_beta_low_alpha",
        axs_titles=["CL (Low Alpha)", "CD (Low Alpha)", "CS (Low Alpha)"],
        legend_location_index=0,
        legend_location="lower left",
    )

    ### 3. literature_polars_beta_high_alpha_correction_vs_no_correction
    # plot_single_row(
    #     results_dir,
    #     data_frames=[
    #         data_VSM_beta_re_56e4_alpha_1195_breukels,
    #         data_VSM_beta_re_56e4_alpha_1195_breukels_stall,
    #         data_VSM_beta_re_56e4_alpha_1195_corrected,
    #         data_VSM_beta_re_56e4_alpha_1195_corrected_stall,
    #         data_WT_beta_re_56e4_alpha_12_9,
    #     ],
    #     labels=[
    #         r"VSM Breukels",
    #         r"VSM Breukels Stall",
    #         r"VSM Corrected",
    #         r"VSM Corrected Stall",
    #         r"WT",
    #     ],
    #     colors=["blue", "blue", "green", "green", "red"],
    #     linestyles=["--", "-", "--", "-", "-"],
    #     markers=["", "", "+", "+", "o"],
    #     confidence_interval=confidence_interval,
    #     file_name=f"literature_polars_beta_alpha_1195_correction_and_stall_effects",
    #     axs_titles=["CL (High Alpha)", "CD (High Alpha)", "CS (High Alpha)"],
    #     legend_location_index=0,
    #     legend_location="lower left",
    # )
    # ### 4. literature_polars_beta_low_alpha_correction_vs_no_correction
    # plot_single_row(
    #     results_dir,
    #     data_frames=[
    #         data_VSM_beta_re_56e4_alpha_675_breukels,
    #         data_VSM_beta_re_56e4_alpha_675_breukels_stall,
    #         data_VSM_beta_re_56e4_alpha_675_corrected,
    #         data_VSM_beta_re_56e4_alpha_675_corrected_stall,
    #         data_WT_beta_re_56e4_alpha_7_7,
    #     ],
    #     labels=[
    #         r"VSM Breukels",
    #         r"VSM Breukels Stall",
    #         r"VSM Corrected",
    #         r"VSM Corrected Stall",
    #         r"WT",
    #     ],
    #     colors=["blue", "blue", "green", "green", "red"],
    #     linestyles=["--", "-", "--", "-", "-"],
    #     markers=["", "", "+", "+", "o"],
    #     confidence_interval=confidence_interval,
    #     file_name=f"literature_polars_beta_alpha_675_correction_and_stall_effects",
    #     axs_titles=["CL (Low Alpha)", "CD (Low Alpha)", "CS (Low Alpha)"],
    #     legend_location_index=0,
    #     legend_location="lower left",
    # )

    ### 5. plotting double rows
    plot_double_row(
        results_dir,
        high_data_frames=[
            data_lebesque_re_100e4_alpha_1195,
            data_VSM_beta_re_56e4_alpha_1195_corrected,
            data_WT_beta_re_56e4_alpha_12_9,
        ],
        high_labels=[
            r"CFD Re = $10\times10^5$ $(\alpha = 13.02^{\circ})$",
            r"VSM Re = $5\times10^5$ (no struts)",
            r"WT Re = $5\times10^5$",
        ],
        high_colors=["black", "blue", "red"],
        high_linestyles=["-", "-", "-"],
        high_markers=["*", ".", "o"],
        low_data_frames=[
            data_VSM_beta_re_56e4_alpha_675_corrected,
            data_WT_beta_re_56e4_alpha_7_7,
        ],
        low_labels=[
            r"VSM Re = $5\times10^5$",
            r"WT Re = $5\times10^5$",
        ],
        low_colors=["blue", "red"],
        low_linestyles=["-", "-"],
        low_markers=[".", "o"],
        file_name="literature_polars_beta_double_alpha",
        confidence_interval=confidence_interval,
        axs_titles_high=["CL (High Alpha)", "CD (High Alpha)", "CS (High Alpha)"],
        axs_titles_low=["CL (Low Alpha)", "CD (Low Alpha)", "CS (Low Alpha)"],
    )

    # printing the averaged alpha values
    from load_balance_analysis.functions_utils import alpha_wind_tunnel_correction

    for alpha in [7.7, 12.9]:
        print(f"\n alpha: {alpha}")
        for cl in data_WT_beta_re_56e4_alpha_7_7["CL"]:
            print(
                f"alpha: {alpha} CL: {cl} corrected alpha: {alpha_wind_tunnel_correction(alpha, cl)}"
            )


def plotting_polars_beta_moments(
    project_dir: str,
    results_dir: str,
    confidence_interval: float,
):
    """
    Plots CMx, CMy, CMz vs beta for high-alpha and low-alpha cases, comparing
    no-correction vs correction for VSM, plus wind tunnel data.
    """
    polar_dir = Path(project_dir) / "processed_data" / "polar_data"
    # 1) Load VSM (High alpha = 12.95 deg) moment data
    # data_VSM_beta_re_56e4_alpha_1195_breukels_moment = pd.read_csv(
    #     Path(polar_dir)
    #     / "VSM_results_beta_sweep_Rey_5.0_alpha_1195_breukels_moment.csv"
    # )
    # data_VSM_beta_re_56e4_alpha_1195_breukels_stall_moment = pd.read_csv(
    #     Path(polar_dir)
    #     / "VSM_results_beta_sweep_Rey_5.0_alpha_1195_breukels_stall_moment.csv"
    # )
    data_VSM_beta_re_56e4_alpha_1195_corrected_moment = pd.read_csv(
        Path(polar_dir)
        / "VSM_results_beta_sweep_Rey_5.1_alpha_1250_corrected_moment.csv"
    )
    # data_VSM_beta_re_56e4_alpha_1195_corrected_stall_moment = pd.read_csv(
    #     Path(polar_dir)
    #     / "VSM_results_beta_sweep_Rey_5.0_alpha_1195_corrected_stall_moment.csv"
    # )

    # 2) Load VSM (Low alpha = 6.75 deg) moment data
    # data_VSM_beta_re_56e4_alpha_675_breukels_moment = pd.read_csv(
    #     Path(polar_dir) / "VSM_results_beta_sweep_Rey_5.0_alpha_740_breukels_moment.csv"
    # )
    # data_VSM_beta_re_56e4_alpha_675_breukels_stall_moment = pd.read_csv(
    #     Path(polar_dir)
    #     / "VSM_results_beta_sweep_Rey_5.0_alpha_740_breukels_stall_moment.csv"
    # )
    data_VSM_beta_re_56e4_alpha_675_corrected_moment = pd.read_csv(
        Path(polar_dir)
        / "VSM_results_beta_sweep_Rey_5.0_alpha_740_corrected_moment.csv"
    )
    # data_VSM_beta_re_56e4_alpha_675_corrected_stall_moment = pd.read_csv(
    #     Path(polar_dir)
    #     / "VSM_results_beta_sweep_Rey_5.0_alpha_740_corrected_stall_moment.csv"
    # )

    # 3) Load Wind Tunnel moment data
    #    (Assuming you saved them as "V3_CMx_CMy_CMz_beta_sweep_alpha_7_7_..." etc.)
    data_WT_beta_re_56e4_alpha_7_7_moment = pd.read_csv(
        Path(project_dir)
        / "processed_data"
        / "polar_data"
        / "V3_CMx_CMy_CMz_beta_sweep_alpha_7.7_WindTunnel_Poland_2025_Rey_560e4.csv"
    )
    data_WT_beta_re_56e4_alpha_12_9_moment = pd.read_csv(
        Path(project_dir)
        / "processed_data"
        / "polar_data"
        / "V3_CMx_CMy_CMz_beta_sweep_alpha_12.9_WindTunnel_Poland_2025_Rey_560e4.csv"
    )

    # ## correcting the beta
    # data_WT_beta_re_56e4_alpha_12_9_moment["beta"] = (
    #     -data_WT_beta_re_56e4_alpha_12_9_moment["beta"]
    # )
    # data_WT_beta_re_56e4_alpha_7_7_moment["beta"] = (
    #     -data_WT_beta_re_56e4_alpha_7_7_moment["beta"]
    # )

    # 1. moment_literature_polars_beta_high_alpha_correction_vs_no_correction
    # plot_single_row(
    #     results_dir=results_dir,
    #     data_frames=[
    #         data_VSM_beta_re_56e4_alpha_675_breukels_moment,
    #         data_VSM_beta_re_56e4_alpha_675_breukels_stall_moment,
    #         data_VSM_beta_re_56e4_alpha_675_corrected_moment,
    #         data_VSM_beta_re_56e4_alpha_675_corrected_stall_moment,
    #         data_WT_beta_re_56e4_alpha_7_7_moment,
    #     ],
    #     labels=[
    #         r"VSM Breukels",
    #         r"VSM Breukels Stall",
    #         r"VSM Corrected",
    #         r"VSM Corrected Stall",
    #         r"WT",
    #     ],
    #     colors=["blue", "blue", "green", "green", "red"],
    #     linestyles=["--", "-", "--", "-", "-"],
    #     markers=["s", "s", "+", "+", "o"],
    #     file_name="moment_literature_polars_beta_alpha_7_7_correction_and_stall_effect",
    #     confidence_interval=confidence_interval,
    #     axs_titles=["CMx (High Alpha)", "CMy (High Alpha)", "CMz (High Alpha)"],
    #     legend_location_index=0,
    #     legend_location="lower left",
    #     variables_to_plot=["CMx", "CMy", "CMz"],  # <--- Key difference
    #     # ylim=[[-0.2, 0.5], [-0.2, 0.5], [-0.3, 0.6]],
    # )

    # # 2. moment_literature_polars_beta_low_alpha_correction_vs_no_correction
    # plot_single_row(
    #     results_dir=results_dir,
    #     data_frames=[
    #         data_VSM_beta_re_56e4_alpha_1195_breukels_moment,
    #         data_VSM_beta_re_56e4_alpha_1195_breukels_stall_moment,
    #         data_VSM_beta_re_56e4_alpha_1195_corrected_moment,
    #         data_VSM_beta_re_56e4_alpha_1195_corrected_stall_moment,
    #         data_WT_beta_re_56e4_alpha_12_9_moment,
    #     ],
    #     labels=[
    #         r"VSM Breukels",
    #         r"VSM Breukels Stall",
    #         r"VSM Corrected",
    #         r"VSM Corrected Stall",
    #         r"WT",
    #     ],
    #     colors=["blue", "blue", "green", "green", "red"],
    #     linestyles=["--", "-", "--", "-", "-"],
    #     markers=["", "", "+", "+", "o"],
    #     file_name="moment_literature_polars_beta_alpha_12_95_correction_and_stall_effect",
    #     confidence_interval=confidence_interval,
    #     axs_titles=["CMx (Low Alpha)", "CMy (Low Alpha)", "CMz (Low Alpha)"],
    #     legend_location_index=0,
    #     legend_location="lower left",
    #     variables_to_plot=["CMx", "CMy", "CMz"],  # <--- Key difference
    #     # ylim=[[-0.2, 0.5], [-0.2, 0.5], [-0.3, 0.6]],
    # )

    # # 3. moment_literature_polars_beta_high_alpha
    # plot_single_row(
    #     results_dir=results_dir,
    #     data_frames=[
    #         data_VSM_beta_re_56e4_alpha_1195_corrected_moment,
    #         data_WT_beta_re_56e4_alpha_12_9_moment,
    #     ],
    #     labels=[
    #         r"VSM $\mathrm{Re} = 5\times10^5$ (no struts)",
    #         r"WT $\mathrm{Re} = $5\times10^5$",
    #     ],
    #     colors=["blue", "red"],
    #     linestyles=["-", "-"],
    #     markers=[".", "o"],
    #     file_name="moment_literature_polars_beta_high_alpha",
    #     confidence_interval=confidence_interval,
    #     axs_titles=["CMx (High Alpha) moment", "CMy (High Alpha)", "CMz (High Alpha)"],
    #     legend_location_index=0,
    #     legend_location="lower left",
    #     variables_to_plot=["CMx", "CMy", "CMz"],  # <--- Key difference
    #     # ylim=[[-0.2, 0.5], [-0.2, 0.5], [-0.3, 0.6]],
    # )

    # 4. moment_literature_polars_beta_low_alpha
    plot_single_row(
        results_dir=results_dir,
        data_frames=[
            data_VSM_beta_re_56e4_alpha_675_corrected_moment,
            data_WT_beta_re_56e4_alpha_7_7_moment,
        ],
        labels=[
            r"VSM $\mathrm{Re} = 5\times10^5$ (no struts)",
            r"WT $\mathrm{Re}e = 5\times10^5$",
        ],
        colors=["blue", "red"],
        linestyles=["-", "-"],
        markers=[".", "o"],
        file_name="moment_literature_polars_beta_low_alpha",
        confidence_interval=confidence_interval,
        axs_titles=["CMx (Low Alpha) moment", "CMy (Low Alpha)", "CMz (Low Alpha)"],
        legend_location_index=0,
        legend_location="lower left",
        variables_to_plot=["CMx", "CMy", "CMz"],  # <--- Key difference
        # ylim=[[-0.2, 0.5], [-0.2, 0.5], [-0.3, 0.6]],
    )

    # ### 5. plotting double rows
    # plot_double_row(
    #     results_dir,
    #     high_data_frames=[
    #         data_VSM_beta_re_56e4_alpha_1195_corrected_moment,
    #         data_WT_beta_re_56e4_alpha_12_9_moment,
    #     ],
    #     high_labels=[
    #         r"VSM $\mathrm{Re} = 5\times10^5$ (no struts)",
    #         r"WT $\mathrm{Re} = 5\times10^5$",
    #     ],
    #     high_colors=["blue", "red"],
    #     high_linestyles=["-", "-"],
    #     high_markers=[".", "o"],
    #     low_data_frames=[
    #         data_VSM_beta_re_56e4_alpha_675_corrected_moment,
    #         data_WT_beta_re_56e4_alpha_7_7_moment,
    #     ],
    #     low_labels=[
    #         r"VSM $\mathrm{Re} = 5\times10^5$ (no struts)",
    #         r"WT $\mathrm{Re} = 5\times10^5$",
    #     ],
    #     low_colors=["blue", "red"],
    #     low_linestyles=["-", "-"],
    #     low_markers=[".", "o"],
    #     file_name="moment_literature_polars_beta_double_alpha",
    #     confidence_interval=confidence_interval,
    #     axs_titles_high=["CMx (High Alpha)", "CMy (High Alpha)", "CMz (High Alpha)"],
    #     axs_titles_low=["CMx (Low Alpha)", "CMy (Low Alpha)", "CMz (Low Alpha)"],
    #     variables_to_plot=["CMx", "CMy", "CMz"],
    # )
