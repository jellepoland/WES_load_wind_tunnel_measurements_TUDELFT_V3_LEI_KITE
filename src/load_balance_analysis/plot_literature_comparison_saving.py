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


def raw_df_to_mean_and_ci(
    df: pd.DataFrame,
    variable_col: str,
    coefficients: list,
    confidence_interval: float,
    max_lag: int,
) -> pd.DataFrame:

    # Initialize an empty list to hold each unique group's results
    all_data_calculated = []

    print(f"variable_col: {variable_col}")
    # Loop through each unique value in variable_col
    for col in df[variable_col].unique():
        # print(f"col: {col}")
        data = df[df[variable_col] == col]

        # Initialize a dictionary for the current group's results
        data_calculated = {}

        # HAC Newey-West Confidence Interval calculation for each coefficient
        for coeff in coefficients:
            # print(f"coeff: {coeff}")
            mean_value = data[coeff].mean()
            ci = hac_newey_west_confidence_interval(
                data[coeff], confidence_interval=confidence_interval, max_lag=max_lag
            )

            # Create entries in the dictionary with dynamic column names
            data_calculated[f"{coeff}_mean"] = mean_value
            data_calculated[f"{coeff}_ci"] = ci

        # Add the variable_col value to the dictionary
        data_calculated[variable_col] = col

        # Append the current group's results to the list
        all_data_calculated.append(data_calculated)

    # Transform the list of dictionaries into a DataFrame
    data_calculated_df = pd.DataFrame(all_data_calculated)

    # sorting the df by the variable col
    data_calculated_df = data_calculated_df.sort_values(by=variable_col)

    return data_calculated_df


def saving_alpha_and_beta_sweeps(
    project_dir: str,
    confidence_interval: float,
    max_lag: int,
    name_appendix: str = "",
):
    """
    Reads wind tunnel CSV data for alpha sweeps (beta=0, multiple velocities)
    and beta sweeps (alpha ~ 6.8 or 11.9 deg), computes mean + confidence intervals,
    and saves out separate files:
      - A file with CL, CD, (CS) vs alpha or beta.
      - A file with CMx, CMy, CMz vs alpha or beta.
    """

    ### Part 1) ALPHA sweeps
    vw_values = [5, 20]
    beta_value = 0
    folders_and_files = [(f"beta_{beta_value}", f"vw_{vw}.csv") for vw in vw_values]
    print(f"folders_and_files: {folders_and_files}")

    data_windtunnel = []

    for folder_name, file_name in folders_and_files:
        path_to_csv = (
            Path(project_dir)
            / "processed_data"
            / "normal_csv"
            / folder_name
            / file_name
        )
        df_all_values = pd.read_csv(path_to_csv)

        # A) Forces: C_L, C_D
        print(f"\nwill now go into raw_df_to_mean_and_ci")
        df_forces = raw_df_to_mean_and_ci(
            df_all_values,
            variable_col="aoa_kite",
            coefficients=["C_L", "C_D"],
            confidence_interval=confidence_interval,
            max_lag=max_lag,
        )
        # df_forces has columns: ["C_L_mean","C_L_ci","C_D_mean","C_D_ci","aoa_kite"]

        # B) Moments: C_roll, C_pitch, C_yaw
        df_moments = raw_df_to_mean_and_ci(
            df_all_values,
            variable_col="aoa_kite",
            coefficients=["C_roll", "C_pitch", "C_yaw"],
            confidence_interval=confidence_interval,
            max_lag=max_lag,
        )
        # df_moments has columns: ["C_roll_mean","C_roll_ci","C_pitch_mean","C_pitch_ci","C_yaw_mean","C_yaw_ci","aoa_kite"]

        # Rename them so the user sees CM_x_mean, etc.
        df_moments.columns = [
            "CM_x_mean",
            "CM_x_ci",
            "CM_y_mean",
            "CM_y_ci",
            "CM_z_mean",
            "CM_z_ci",
            "aoa_kite",
        ]

        # Store this tuple (forces, moments)
        data_windtunnel.append((df_forces, df_moments))

    # We only use the second one (index=1) in your code
    df_forces_alpha = data_windtunnel[1][0]
    df_moments_alpha = data_windtunnel[1][1]

    # df_forces_alpha => rename columns to [CL, CL_ci, CD, CD_ci, aoa]
    # ( currently = ["C_L_mean","C_L_ci","C_D_mean","C_D_ci","aoa_kite"] )
    col_names_forces = ["CL", "CL_ci", "CD", "CD_ci", "aoa"]
    df_forces_alpha.columns = col_names_forces

    # df_moments_alpha => rename columns to [CMx, CMx_ci, CMy, CMy_ci, CMz, CMz_ci, aoa]
    # ( currently = ["CM_x_mean","CM_x_ci","CM_y_mean","CM_y_ci","CM_z_mean","CM_z_ci","aoa_kite"] )
    col_names_moments = ["CMx", "CMx_ci", "CMy", "CMy_ci", "CMz", "CMz_ci", "aoa"]
    df_moments_alpha.columns = col_names_moments

    ## wind tunnel corrections
    print(f"\n=== UNCORRECTED ===")
    print(f"df_forces_alpha: {df_forces_alpha.columns}")
    print(f"df_moments_alpha: {df_moments_alpha.columns}")
    print(f'df_forces_alpha: {df_forces_alpha["aoa"]}')
    print(f"df_moments_alpha: {df_moments_alpha['aoa']}")

    print(f"\n=== CORRECTED ===")
    df_forces_alpha = apply_angle_wind_tunnel_corrections_to_df(df_forces_alpha)
    df_moments_alpha["aoa"] = df_forces_alpha["aoa"]
    print(f'df_forces_alpha: {df_forces_alpha["aoa"]}')
    print(f"df_moments_alpha: {df_moments_alpha['aoa']}")
    # Save them out
    df_forces_alpha.to_csv(
        Path(project_dir)
        / "processed_data"
        / "polar_data"
        / f"V3_CL_CD_CS_alpha_sweep_for_beta_0_WindTunnel_Poland_2025_Rey_560e4{name_appendix}.csv",
        index=False,
    )
    df_moments_alpha.to_csv(
        Path(project_dir)
        / "processed_data"
        / "polar_data"
        / f"V3_CMx_CMy_CMz_alpha_sweep_for_beta_0_WindTunnel_Poland_2025_Rey_560e4{name_appendix}.csv",
        index=False,
    )

    ### Part 2) BETA sweeps
    vw = 20
    alpha_high = 12.9
    alpha_low = 7.7

    # -- alpha_high
    folder_name = f"alpha_{alpha_high}"
    file_name = f"vw_{vw}.csv"
    path_to_csv = (
        Path(project_dir) / "processed_data" / "normal_csv" / folder_name / file_name
    )
    df_all_values = pd.read_csv(path_to_csv)

    # A) Force coefficients: [C_L, C_D, C_S]
    data_WT_beta_high_forces = raw_df_to_mean_and_ci(
        df_all_values,
        variable_col="sideslip",
        coefficients=["C_L", "C_D", "C_S"],
        confidence_interval=confidence_interval,
        max_lag=max_lag,
    )
    # B) Moment coefficients: [C_Mx, C_My, C_Mz]
    data_WT_beta_high_moments = raw_df_to_mean_and_ci(
        df_all_values,
        variable_col="sideslip",
        coefficients=["C_roll", "C_pitch", "C_yaw"],
        confidence_interval=confidence_interval,
        max_lag=max_lag,
    )

    # data_WT_beta_high_forces => rename columns
    col_names_forces_beta = ["CL", "CL_ci", "CD", "CD_ci", "CS", "CS_ci", "beta"]
    data_WT_beta_high_forces.columns = col_names_forces_beta

    # data_WT_beta_high_moments => rename columns
    col_names_moments_beta = ["CMx", "CMx_ci", "CMy", "CMy_ci", "CMz", "CMz_ci", "beta"]
    data_WT_beta_high_moments.columns = col_names_moments_beta

    # wind tunnel corrections
    data_WT_beta_high_forces = apply_angle_wind_tunnel_corrections_to_df(
        data_WT_beta_high_forces
    )
    data_WT_beta_high_moments["beta"] = data_WT_beta_high_forces["beta"]

    # Save them
    data_WT_beta_high_forces.to_csv(
        Path(project_dir)
        / "processed_data"
        / "polar_data"
        / f"V3_CL_CD_CS_beta_sweep_alpha_{alpha_high}_WindTunnel_Poland_2025_Rey_560e4{name_appendix}.csv",
        index=False,
    )
    data_WT_beta_high_moments.to_csv(
        Path(project_dir)
        / "processed_data"
        / "polar_data"
        / f"V3_CMx_CMy_CMz_beta_sweep_alpha_{alpha_high}_WindTunnel_Poland_2025_Rey_560e4{name_appendix}.csv",
        index=False,
    )

    # -- alpha_low
    folder_name = f"alpha_{alpha_low}"
    file_name = f"vw_{vw}.csv"
    path_to_csv = (
        Path(project_dir) / "processed_data" / "normal_csv" / folder_name / file_name
    )
    df_all_values = pd.read_csv(path_to_csv)

    data_WT_beta_low_forces = raw_df_to_mean_and_ci(
        df_all_values,
        variable_col="sideslip",
        coefficients=["C_L", "C_D", "C_S"],
        confidence_interval=confidence_interval,
        max_lag=max_lag,
    )
    data_WT_beta_low_moments = raw_df_to_mean_and_ci(
        df_all_values,
        variable_col="sideslip",
        coefficients=["C_roll", "C_pitch", "C_yaw"],
        confidence_interval=confidence_interval,
        max_lag=max_lag,
    )

    # changing column names
    data_WT_beta_low_forces.columns = col_names_forces_beta
    data_WT_beta_low_moments.columns = col_names_moments_beta

    # wind tunnel corrections
    data_WT_beta_low_forces = apply_angle_wind_tunnel_corrections_to_df(
        data_WT_beta_low_forces
    )
    data_WT_beta_low_moments["beta"] = data_WT_beta_low_forces["beta"]

    # Save them
    data_WT_beta_low_forces.to_csv(
        Path(project_dir)
        / "processed_data"
        / "polar_data"
        / f"V3_CL_CD_CS_beta_sweep_alpha_{alpha_low}_WindTunnel_Poland_2025_Rey_560e4{name_appendix}.csv",
        index=False,
    )

    data_WT_beta_low_moments.to_csv(
        Path(project_dir)
        / "processed_data"
        / "polar_data"
        / f"V3_CMx_CMy_CMz_beta_sweep_alpha_{alpha_low}_WindTunnel_Poland_2025_Rey_560e4{name_appendix}.csv",
        index=False,
    )
