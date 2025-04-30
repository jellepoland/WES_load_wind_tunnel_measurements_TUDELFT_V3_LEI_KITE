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


def plotting_polars_alpha(
    project_dir: str,
    results_dir: str,
    confidence_interval: float,
):

    # VSM and Lebesque data
    path_to_csv_VSM_alpha_re_56e4 = (
        Path(project_dir)
        / "processed_data"
        / "polar_data"
        / f"VSM_results_alpha_sweep_Rey_5.0_corrected.csv"
    )
    data_VSM_alpha_re_56e4 = pd.read_csv(path_to_csv_VSM_alpha_re_56e4)

    data_CFD_Vire2020_5e5 = pd.read_csv(
        Path(project_dir)
        / "data"
        / "CFD_polar_data"
        / "CFD_V3_CL_CD_RANS_Vire2020_Rey_50e4.csv"
    )
    data_CFD_Vire2021_10e5 = pd.read_csv(
        Path(project_dir)
        / "data"
        / "CFD_polar_data"
        / "CFD_RANS_Rey_10e5_Poland2025_alpha_sweep_beta_0.csv"
    )

    # Get wind tunnel data
    path_to_csv = (
        Path(project_dir)
        / "processed_data"
        / "polar_data"
        / "V3_CL_CD_CS_alpha_sweep_for_beta_0_WindTunnel_Poland_2025_Rey_560e4.csv"
    )
    data_windtunnel = pd.read_csv(path_to_csv)

    # Data frames, labels, colors, and linestyles
    data_frame_list = [
        data_CFD_Vire2021_10e5,
        data_CFD_Vire2020_5e5,
        data_VSM_alpha_re_56e4,
        data_windtunnel,
    ]
    labels = [
        rf"CFD Re = $10\times10^5$ (Struts)",
        rf"CFD Re = $5\times10^5$",
        rf"VSM Re = $5\times10^5$",
        rf"WT Re = $5\times10^5$",
        # rf"Polars Uri",
        # rf"CFD Re = $10\times10^5$ -1.02",
        # rf"CFD Re = $10\times10^5$ +1.02",
    ]
    colors = ["black", "black", "blue", "red", "green", "purple"]
    linestyles = ["--", "-", "-", "-", "dashdot", "dashdot"]
    markers = ["*", "*", ".", "o", "x", "x"]
    fmt_wt = "o"

    # Plot CL, CD, and CL/CD in subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    aoa_col = f"aoa"
    cl_col = f"CL"
    cd_col = f"CD"

    for i, (data_frame, label, color, linestyle, marker) in enumerate(
        zip(data_frame_list, labels, colors, linestyles, markers)
    ):
        plot_on_ax(
            axs[0],
            data_frame[aoa_col],
            data_frame[cl_col],
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            is_with_grid=True,
        )
        plot_on_ax(
            axs[1],
            data_frame[aoa_col],
            data_frame[cd_col],
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            is_with_grid=True,
        )
        plot_on_ax(
            axs[2],
            data_frame[aoa_col],
            data_frame[cl_col] / data_frame[cd_col],
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            is_with_grid=True,
        )
        # adding the confidence interval
        if i == 3:
            alpha = 0.2
            axs[0].fill_between(
                data_frame[aoa_col],
                data_frame[cl_col] - data_frame["CL_ci"],
                data_frame[cl_col] + data_frame["CL_ci"],
                color=color,
                alpha=alpha,
                label=f"WT CI of {confidence_interval}\%",
            )
            axs[1].fill_between(
                data_frame[aoa_col],
                data_frame[cd_col] - data_frame["CD_ci"],
                data_frame[cd_col] + data_frame["CD_ci"],
                color=color,
                alpha=alpha,
                label=f"WT CI of {confidence_interval}\%",
            )
            axs[2].fill_between(
                data_frame[aoa_col],
                data_frame[cl_col] / data_frame[cd_col]
                - data_frame["CL_ci"] / data_frame["CD_ci"],
                data_frame[cl_col] / data_frame[cd_col]
                + data_frame["CL_ci"] / data_frame["CD_ci"],
                color=color,
                alpha=alpha,
                label=f"WT CI of {confidence_interval}\%",
            )

    axs[0].set_xlim(-12, 25)
    axs[1].set_xlim(-12, 25)
    axs[2].set_xlim(-12, 25)

    # Format axes
    axs[0].set_xlabel(x_axis_labels["alpha"])
    axs[0].set_ylabel(y_axis_labels["CL"])
    # axs[0].grid()

    axs[1].set_xlabel(x_axis_labels["alpha"])
    axs[1].set_ylabel(y_axis_labels["CD"])
    axs[1].legend(loc="upper left")
    # axs[1].grid()

    axs[2].set_xlabel(x_axis_labels["alpha"])
    axs[2].set_ylabel(y_axis_labels["L/D"])
    # axs[2].grid()

    # Save the plot
    plt.tight_layout()
    file_name = f"literature_polars_alpha"
    saving_pdf_and_pdf_tex(results_dir, file_name)


# def plotting_polars_alpha_correction_comparison(
#     project_dir: str,
#     results_dir: str,
#     confidence_interval: float,
# ):
#     polar_dir = Path(project_dir) / "processed_data" / "polar_data"
#     # VSM data
#     data_VSM_alpha_re_56e4_breukels = pd.read_csv(
#         Path(polar_dir) / f"VSM_results_alpha_sweep_Rey_5.6_breukels.csv"
#     )
#     data_VSM_alpha_re_56e4_breukels_stall = pd.read_csv(
#         Path(polar_dir) / f"VSM_results_alpha_sweep_Rey_5.6_breukels_stall.csv"
#     )
#     data_VSM_alpha_re_56e4_corrected = pd.read_csv(
#         Path(polar_dir) / f"VSM_results_alpha_sweep_Rey_5.6_corrected.csv"
#     )
#     data_VSM_alpha_re_56e4_corrected_stall = pd.read_csv(
#         Path(polar_dir) / f"VSM_results_alpha_sweep_Rey_5.6_corrected_stall.csv"
#     )

#     # Get wind tunnel data
#     path_to_csv = (
#         Path(polar_dir)
#         / "V3_CL_CD_CS_alpha_sweep_for_beta_0_WindTunnel_Poland_2025_Rey_560e4.csv"
#     )
#     data_windtunnel = pd.read_csv(path_to_csv)

#     data_CFD_Vire2020_5e5 = pd.read_csv(
#         Path(polar_dir) / "CFD_V3_CL_CD_RANS_Vire2020_Rey_50e4.csv"
#     )
#     data_CFD_Vire2021_10e5 = pd.read_csv(
#         Path(polar_dir) / "CFD_V3_CL_CD_RANS_Lebesque_2024_Rey_100e4.csv"
#     )

#     # Data frames, labels, colors, and linestyles
#     data_frame_list = [
#         data_CFD_Vire2020_5e5,
#         data_CFD_Vire2021_10e5,
#         data_VSM_alpha_re_56e4_breukels,
#         data_VSM_alpha_re_56e4_breukels_stall,
#         data_VSM_alpha_re_56e4_corrected,
#         data_VSM_alpha_re_56e4_corrected_stall,
#         data_windtunnel,
#     ]
#     labels = [
#         rf"CFD Re = $5\times10^5$ (No struts)",
#         rf"CFD Re = $10\times10^5$",
#         rf"VSM Breukels Re = $5.6\times10^5$",
#         rf"VSM Breukels Stall Re = $5.6\times10^5$",
#         rf"VSM Corrected Re = $5.6\times10^5$",
#         rf"VSM Corrected Stall Re = $5.6\times10^5$",
#         rf"WT Re = $5.6\times10^5$",
#     ]
#     colors = ["black", "black", "blue", "blue", "green", "green", "red"]
#     linestyles = ["--", "-", "--", "-", "--", "-", "-"]
#     markers = ["*", "*", "s", "s", "+", "+", "o"]
#     fmt_wt = "o"

#     # Plot CL, CD, and CL/CD in subplots
#     fig, axs = plt.subplots(1, 3, figsize=(15, 5))

#     aoa_col = f"aoa"
#     cl_col = f"CL"
#     cd_col = f"CD"

#     for i, (data_frame, label, color, linestyle, marker) in enumerate(
#         zip(data_frame_list, labels, colors, linestyles, markers)
#     ):
#         plot_on_ax(
#             axs[0],
#             data_frame[aoa_col],
#             data_frame[cl_col],
#             label=label,
#             color=color,
#             linestyle=linestyle,
#             marker=marker,
#             is_with_grid=True,
#         )
#         plot_on_ax(
#             axs[1],
#             data_frame[aoa_col],
#             data_frame[cd_col],
#             label=label,
#             color=color,
#             linestyle=linestyle,
#             marker=marker,
#             is_with_grid=True,
#         )
#         plot_on_ax(
#             axs[2],
#             data_frame[aoa_col],
#             data_frame[cl_col] / data_frame[cd_col],
#             label=label,
#             color=color,
#             linestyle=linestyle,
#             marker=marker,
#             is_with_grid=True,
#         )
#         # adding the confidence interval
#         if "WT" in label:
#             alpha = 0.2
#             axs[0].fill_between(
#                 data_frame[aoa_col],
#                 data_frame[cl_col] - data_frame["CL_ci"],
#                 data_frame[cl_col] + data_frame["CL_ci"],
#                 color=color,
#                 alpha=alpha,
#                 label=f"WT CI of {confidence_interval}%",
#             )
#             axs[1].fill_between(
#                 data_frame[aoa_col],
#                 data_frame[cd_col] - data_frame["CD_ci"],
#                 data_frame[cd_col] + data_frame["CD_ci"],
#                 color=color,
#                 alpha=alpha,
#                 label=f"WT CI of {confidence_interval}%",
#             )
#             axs[2].fill_between(
#                 data_frame[aoa_col],
#                 data_frame[cl_col] / data_frame[cd_col]
#                 - data_frame["CL_ci"] / data_frame["CD_ci"],
#                 data_frame[cl_col] / data_frame[cd_col]
#                 + data_frame["CL_ci"] / data_frame["CD_ci"],
#                 color=color,
#                 alpha=alpha,
#                 label=f"WT CI of {confidence_interval}%",
#             )

#     axs[0].set_xlim(-13, 24)
#     axs[1].set_xlim(-13, 24)
#     axs[2].set_xlim(-13, 24)

#     # Format axes
#     axs[0].set_xlabel(x_axis_labels["alpha"])
#     axs[0].set_ylabel(y_axis_labels["CL"])
#     # axs[0].grid()

#     axs[1].set_xlabel(x_axis_labels["alpha"])
#     axs[1].set_ylabel(y_axis_labels["CD"])
#     axs[1].legend(loc="upper left")
#     # axs[1].grid()
#     axs[1].set_ylim(0, 0.6)

#     axs[2].set_xlabel(x_axis_labels["alpha"])
#     axs[2].set_ylabel(y_axis_labels["L/D"])
#     # axs[2].grid()

#     # Save the plot
#     plt.tight_layout()
#     file_name = f"literature_polars_alpha_correction_and_stall_effects"
#     saving_pdf_and_pdf_tex(results_dir, file_name)


# def plotting_polars_alpha_moments_correction_comparison(
#     project_dir: str,
#     results_dir: str,
#     confidence_interval: float,
# ):
#     """
#     Plots CMx, CMy, CMz as a function of alpha, comparing:
#       - VSM no-correction
#       - VSM corrected
#       - Wind tunnel data
#     for Re ~ 5.6e5, alpha sweep at beta=0.

#     Filenames assume you have:
#       VSM_results_alpha_sweep_Re_5.6_no_correction_moment.csv
#       VSM_results_alpha_sweep_Re_5.6_moment.csv
#       V3_CMx_CMy_CMz_alpha_sweep_for_beta_0_WindTunnel_Poland_2025_Rey_560e4.csv
#     or similar in your 'polar_data' folder.
#     """
#     polar_dir = Path(project_dir) / "processed_data" / "polar_data"

#     # 1) Load VSM
#     data_VSM_alpha_re_56e4_breukels_moment = pd.read_csv(
#         Path(polar_dir) / "VSM_results_alpha_sweep_Rey_5.6_breukels_moment.csv"
#     )
#     data_VSM_alpha_re_56e4_breukels_stall_moment = pd.read_csv(
#         Path(polar_dir) / "VSM_results_alpha_sweep_Rey_5.6_breukels_stall_moment.csv"
#     )
#     data_VSM_alpha_re_56e4_corrected_moment = pd.read_csv(
#         Path(polar_dir) / "VSM_results_alpha_sweep_Rey_5.6_corrected_moment.csv"
#     )
#     data_VSM_alpha_re_56e4_corrected_stall_moment = pd.read_csv(
#         Path(polar_dir) / "VSM_results_alpha_sweep_Rey_5.6_corrected_stall_moment.csv"
#     )

#     # 3) Load wind tunnel (WT) moment data
#     path_to_csv_WT_moment = (
#         Path(project_dir)
#         / "processed_data"
#         / "polar_data"
#         / "V3_CMx_CMy_CMz_alpha_sweep_for_beta_0_WindTunnel_Poland_2025_Rey_560e4.csv"
#     )
#     data_WT_alpha_moment = pd.read_csv(path_to_csv_WT_moment)

#     # Put them in a list for easy iteration
#     data_frame_list = [
#         data_VSM_alpha_re_56e4_breukels_moment,
#         data_VSM_alpha_re_56e4_breukels_stall_moment,
#         data_VSM_alpha_re_56e4_corrected_moment,
#         data_VSM_alpha_re_56e4_corrected_stall_moment,
#         data_WT_alpha_moment,
#     ]
#     labels = [
#         r"VSM Breukels $\mathrm{Re} = 5.6\times10^5$",
#         r"VSM Breukels Stall $\mathrm{Re} = 5.6\times10^5$",
#         r"VSM Corrected $\mathrm{Re} = 5.6\times10^5$",
#         r"VSM Corrected Stall$\mathrm{Re} = 5.6\times10^5$",
#         r"WT $\mathrm{Re} = 5.6\times10^5$",
#     ]
#     colors = ["blue", "blue", "green", "green", "red"]
#     linestyles = ["--", "-", "--", "-", "-"]
#     markers = ["s", "s", "+", "+", "o"]

#     # 4) Create subplots: CMx, CMy, CMz
#     fig, axs = plt.subplots(1, 3, figsize=(15, 5))

#     # Columns used in each DataFrame
#     alpha_col = "aoa"
#     cmx_col = "CMx"
#     cmy_col = "CMy"
#     cmz_col = "CMz"

#     # Also, the columns that store the confidence intervals (for the WT data)
#     # e.g. "CMx_ci", "CMy_ci", "CMz_ci"
#     cmx_ci_col = "CMx_ci"
#     cmy_ci_col = "CMy_ci"
#     cmz_ci_col = "CMz_ci"

#     # 5) Plot each dataset
#     for i, (data_frame, label, color, linestyle, marker) in enumerate(
#         zip(data_frame_list, labels, colors, linestyles, markers)
#     ):
#         # Plot CMx vs alpha
#         plot_on_ax(
#             axs[0],
#             data_frame[alpha_col],
#             data_frame[cmx_col],
#             label=label,
#             color=color,
#             linestyle=linestyle,
#             marker=marker,
#             is_with_grid=True,
#         )
#         # Plot CMy vs alpha
#         plot_on_ax(
#             axs[1],
#             data_frame[alpha_col],
#             data_frame[cmy_col],
#             label=label,
#             color=color,
#             linestyle=linestyle,
#             marker=marker,
#             is_with_grid=True,
#         )
#         # Plot CMz vs alpha
#         plot_on_ax(
#             axs[2],
#             data_frame[alpha_col],
#             data_frame[cmz_col],
#             label=label,
#             color=color,
#             linestyle=linestyle,
#             marker=marker,
#             is_with_grid=True,
#         )

#         # If this is the WT dataset (i==2), fill confidence intervals
#         if "WT" in label and cmx_ci_col in data_frame.columns:
#             alpha_shade = 0.2
#             # CMx
#             axs[0].fill_between(
#                 data_frame[alpha_col],
#                 data_frame[cmx_col] - data_frame[cmx_ci_col],
#                 data_frame[cmx_col] + data_frame[cmx_ci_col],
#                 color=color,
#                 alpha=alpha_shade,
#                 label=f"WT CI of {confidence_interval}%",
#             )
#             # CMy
#             axs[1].fill_between(
#                 data_frame[alpha_col],
#                 data_frame[cmy_col] - data_frame[cmy_ci_col],
#                 data_frame[cmy_col] + data_frame[cmy_ci_col],
#                 color=color,
#                 alpha=alpha_shade,
#                 label=f"WT CI of {confidence_interval}%",
#             )
#             # CMz
#             axs[2].fill_between(
#                 data_frame[alpha_col],
#                 data_frame[cmz_col] - data_frame[cmz_ci_col],
#                 data_frame[cmz_col] + data_frame[cmz_ci_col],
#                 color=color,
#                 alpha=alpha_shade,
#                 label=f"WT CI of {confidence_interval}%",
#             )

#     # 6) Format axes, add labels
#     # X-limits (optional)
#     axs[0].set_xlim(-13, 24)
#     axs[1].set_xlim(-13, 24)
#     axs[2].set_xlim(-13, 24)

#     # X-axis
#     axs[0].set_xlabel(x_axis_labels["alpha"])
#     axs[1].set_xlabel(x_axis_labels["alpha"])
#     axs[2].set_xlabel(x_axis_labels["alpha"])

#     # Y-axis
#     axs[0].set_ylabel("CMx")
#     axs[1].set_ylabel("CMy")
#     axs[2].set_ylabel("CMz")

#     # Legend on the second subplot (or whichever you like)
#     axs[0].legend(loc="upper left")

#     # 7) Save the figure
#     plt.tight_layout()
#     file_name = "literature_polars_alpha_moments_correction_and_stall_effects"
#     saving_pdf_and_pdf_tex(results_dir, file_name)


def plotting_polars_alpha_moments(
    project_dir: str,
    results_dir: str,
    confidence_interval: float,
):
    """
    Plots CMx, CMy, CMz as a function of alpha, comparing:
      - VSM
      - Wind tunnel data
    for Re ~ 5.6e5, alpha sweep at beta=0.

    Filenames assume you have:
      VSM_results_alpha_sweep_Re_5.6_no_correction_moment.csv
      V3_CMx_CMy_CMz_alpha_sweep_for_beta_0_WindTunnel_Poland_2025_Rey_560e4.csv
    or similar in your 'polar_data' folder.
    """

    # 1) Load VSM (no correction) moment data
    path_to_csv_VSM_alpha_re_56e4_no_correction_moment = (
        Path(project_dir)
        / "processed_data"
        / "polar_data"
        / "VSM_results_alpha_sweep_Rey_5.0_corrected_moment.csv"
    )
    data_VSM_alpha_re_56e4_no_correction_moment = pd.read_csv(
        path_to_csv_VSM_alpha_re_56e4_no_correction_moment
    )

    # 2) Load wind tunnel (WT) moment data
    path_to_csv_WT_moment = (
        Path(project_dir)
        / "processed_data"
        / "polar_data"
        / "V3_CMx_CMy_CMz_alpha_sweep_for_beta_0_WindTunnel_Poland_2025_Rey_560e4.csv"
    )
    data_WT_alpha_moment = pd.read_csv(path_to_csv_WT_moment)

    # Put them in a list for easy iteration
    data_frame_list = [
        data_VSM_alpha_re_56e4_no_correction_moment,
        data_WT_alpha_moment,
    ]
    labels = [
        r"VSM Re = $5\times10^5$",
        r"WT Re = $5\times10^5$",
    ]
    colors = ["blue", "red"]
    linestyles = ["-", "-"]
    markers = [".", "o"]

    # 3) Create subplots: CMx, CMy, CMz
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Columns used in each DataFrame
    alpha_col = "aoa"
    cmx_col = "CMx"
    cmy_col = "CMy"
    cmz_col = "CMz"

    # Also, the columns that store the confidence intervals (for the WT data)
    # e.g. "CMx_ci", "CMy_ci", "CMz_ci"
    cmx_ci_col = "CMx_ci"
    cmy_ci_col = "CMy_ci"
    cmz_ci_col = "CMz_ci"

    # 4) Plot each dataset
    for i, (data_frame, label, color, linestyle, marker) in enumerate(
        zip(data_frame_list, labels, colors, linestyles, markers)
    ):
        # Plot CMx vs alpha
        axs[0].plot(
            data_frame[alpha_col],
            data_frame[cmx_col],
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=5,
            linewidth=1.5,
        )
        # Plot CMy vs alpha
        axs[1].plot(
            data_frame[alpha_col],
            data_frame[cmy_col],
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=5,
            linewidth=1.5,
        )
        # Plot CMz vs alpha
        axs[2].plot(
            data_frame[alpha_col],
            data_frame[cmz_col],
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=5,
            linewidth=1.5,
        )

        # If this is the WT dataset (i==1), fill confidence intervals
        if i == 1 and all(
            col in data_frame.columns for col in [cmx_ci_col, cmy_ci_col, cmz_ci_col]
        ):
            alpha_shade = 0.2
            # CMx
            axs[0].fill_between(
                data_frame[alpha_col],
                data_frame[cmx_col] - data_frame[cmx_ci_col],
                data_frame[cmx_col] + data_frame[cmx_ci_col],
                color=color,
                alpha=alpha_shade,
                label=f"WT CI of {confidence_interval}\%",
            )
            # CMy
            axs[1].fill_between(
                data_frame[alpha_col],
                data_frame[cmy_col] - data_frame[cmy_ci_col],
                data_frame[cmy_col] + data_frame[cmy_ci_col],
                color=color,
                alpha=alpha_shade,
                label=f"WT CI of {confidence_interval}\%",
            )
            # CMz
            axs[2].fill_between(
                data_frame[alpha_col],
                data_frame[cmz_col] - data_frame[cmz_ci_col],
                data_frame[cmz_col] + data_frame[cmz_ci_col],
                color=color,
                alpha=alpha_shade,
                label=f"WT CI of {confidence_interval}\%",
            )

    # 5) Format axes, add labels
    # X-limits (optional)
    for ax in axs:
        ax.set_xlim(-12, 25)

    # Y-limits
    # axs[0].set_ylim(-0.05, 0.05)
    # axs[1].set_ylim(-1, 1)
    # axs[2].set_ylim(-0.05, 0.05)

    # X-axis labels
    for ax in axs:
        ax.set_xlabel(r"$\alpha$ ($^{\circ}$)")

    # Y-axis labels
    axs[0].set_ylabel(y_axis_labels["CMx"])
    axs[1].set_ylabel(y_axis_labels["CMy"])
    axs[2].set_ylabel(y_axis_labels["CMz"])

    # Legend on the second subplot (or whichever you prefer)
    axs[2].legend(loc="upper left")

    # 6) Save the figure
    plt.tight_layout()
    file_name = "moment_literature_polar_alphas"
    saving_pdf_and_pdf_tex(results_dir, file_name)

    # Optionally, display the plot
    plt.show()


#
