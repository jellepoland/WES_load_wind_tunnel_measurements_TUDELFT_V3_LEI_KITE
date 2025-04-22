import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path
from load_balance_analysis.functions_utils import project_dir, save_latex_table
from load_balance_analysis.functions_statistics import (
    hac_newey_west_confidence_interval,
)
from statsmodels.tsa.stattools import acf  # Corrected import
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def analyze_autocorrelation(data, column_name, max_lags=20):
    """
    Create comprehensive autocorrelation analysis plots for time series data.

    Args:
        data: array-like, the time series data to analyze
        column_name: str, name of the column being analyzed
        max_lags: int, maximum number of lags to analyze
    """
    # Create a figure with multiple subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Time Series Plot
    ax1.plot(data, label="Original Data")
    ax1.set_title(f"Time Series: {column_name}")
    ax1.set_xlabel("Time Index")
    ax1.set_ylabel("Value")
    ax1.legend()

    # 2. Autocorrelation Function (ACF) Plot
    plot_acf(
        data,
        lags=max_lags,
        ax=ax2,
        title=f"Autocorrelation Function (ACF): {column_name}",
    )

    # 3. Partial Autocorrelation Function (PACF) Plot
    plot_pacf(
        data,
        lags=max_lags,
        ax=ax3,
        title=f"Partial Autocorrelation (PACF): {column_name}",
    )

    # 4. Lag Plot (t vs t-1)
    ax4.scatter(data[:-1], data[1:], alpha=0.5)
    ax4.set_title(f"Lag Plot (t vs t-1): {column_name}")
    ax4.set_xlabel("Value at t")
    ax4.set_ylabel("Value at t+1")

    plt.tight_layout()
    plt.show()

    # Print numerical autocorrelation values for first few lags
    acf_values = acf(data, nlags=5)
    print(f"\nAutocorrelation values for {column_name}:")
    print("Lag  |  ACF Value")
    print("-" * 20)
    for i, value in enumerate(acf_values):
        print(f"{i:2d}   |  {value:.3f}")


def generate_table(data):
    # Create a DataFrame for LaTeX styling
    table_data = {
        ("Re $\cdot 10^5$", "1.4"): [
            data["1.4"]["C_L"],
            data["1.4"]["C_D"],
            data["1.4"]["C_S"],
            data["1.4"]["C_roll"],
            data["1.4"]["C_pitch"],
            data["1.4"]["C_yaw"],
        ],
        ("Re $\cdot 10^5$", "2.8"): [
            data["2.8"]["C_L"],
            data["2.8"]["C_D"],
            data["2.8"]["C_S"],
            data["2.8"]["C_roll"],
            data["2.8"]["C_pitch"],
            data["2.8"]["C_yaw"],
        ],
        ("Re $\cdot 10^5$", "4.2"): [
            data["4.2"]["C_L"],
            data["4.2"]["C_D"],
            data["4.2"]["C_S"],
            data["4.2"]["C_roll"],
            data["4.2"]["C_pitch"],
            data["4.2"]["C_yaw"],
        ],
        ("Re $\cdot 10^5$", "5.6"): [
            data["5.6"]["C_L"],
            data["5.6"]["C_D"],
            data["5.6"]["C_S"],
            data["5.6"]["C_roll"],
            data["5.6"]["C_pitch"],
            data["5.6"]["C_yaw"],
        ],
    }
    index_labels = [
        r"$\bar{\sigma}_{\textrm{L}}$",
        r"$\bar{\sigma}_{\textrm{L}}$",
        r"$\bar{\sigma}_{\textrm{S}}$",
        r"$\bar{\sigma}_{\textrm{{M,x}}$",
        r"$\bar{\sigma}_{\textrm{{M,y}}$",
        r"$\bar{\sigma}_{\textrm{{M,z}}$",
    ]

    df_table = pd.DataFrame(table_data, index=index_labels)
    df_table.columns = pd.MultiIndex.from_tuples(
        df_table.columns
    )  # Format MultiIndex for column labels
    return df_table


def main(project_dir: str) -> pd.DataFrame:
    # Load and store processed data

    # Load data
    save_csv_dir = Path(project_dir) / "processed_data" / "uncertainty_table"

    df_vw_5 = pd.read_csv(save_csv_dir / "df_vw_5.csv")
    df_vw_10 = pd.read_csv(save_csv_dir / "df_vw_10.csv")
    df_vw_15 = pd.read_csv(save_csv_dir / "df_vw_15.csv")
    df_vw_20 = pd.read_csv(save_csv_dir / "df_vw_20.csv")

    print(df_vw_5.columns)
    # breakpoint()

    df_list = [df_vw_5, df_vw_10, df_vw_15, df_vw_20]
    Reynolds_list = ["1.4", "2.8", "4.2", "5.6"]

    # Dictionary to store data for the table
    table_data = {Re: {} for Re in Reynolds_list}

    table_data_2 = {Re: {} for Re in Reynolds_list}

    for df, Re in zip(df_list, Reynolds_list):
        # Compute relative standard deviation for each coefficient
        for col in ["C_D", "C_S", "C_L", "C_roll", "C_pitch", "C_yaw"]:

            if col in ["C_S", "C_roll", "C_yaw"]:
                # filter the df, only take sideslip values that are positive > 0
                df = df[df["sideslip"] > 0]

            mean_mean = np.abs(df[f"{col}_mean"].mean())
            std_mean = df[f"{col}_std"].mean()
            rel_std_mean = std_mean / mean_mean
            table_data[Re][col] = f"{rel_std_mean:.2f}"

            RSD = df[f"{col}_std"] / df[f"{col}_mean"]
            RSD_mean = np.abs(RSD.mean())
            RSD_std = RSD.std()
            table_data_2[Re][col] = f"{RSD_mean:.2f} ({RSD_std:.2f})"

    # for Re in Reynolds_list:
    #     print(f"\nRe: {Re}")
    #     for key, value in table_data[Re].items():
    #         print(f"{key}: {value}")

    # Generate the table
    table_df = generate_table(table_data)
    save_latex_table(
        table_df,
        Path(project_dir) / "results" / "tables" / "relative_standard_deviation.tex",
    )
    print(f"\n--- Relative standard deviation table ---\n")
    print(table_df.to_string(index=False))
    print(table_df)
    table_df_2 = generate_table(table_data_2)
    print(table_df_2)


# def main(project_dir: str) -> pd.DataFrame:
#     # Load and store processed data

#     # Load data
#     save_csv_dir = Path(project_dir) / "processed_data" / "uncertainty_table"

#     df_vw_5 = pd.read_csv(save_csv_dir / "df_vw_5.csv")
#     df_vw_10 = pd.read_csv(save_csv_dir / "df_vw_10.csv")
#     df_vw_15 = pd.read_csv(save_csv_dir / "df_vw_15.csv")
#     df_vw_20 = pd.read_csv(save_csv_dir / "df_vw_20.csv")

#     df_list = [df_vw_5, df_vw_10, df_vw_15, df_vw_20]
#     Reynolds_list = ["1.4", "2.8", "4.2", "5.6"]

#     # Dictionary to store data for the table
#     table_data = {Re: {} for Re in Reynolds_list}

#     for df, Re in zip(df_list, Reynolds_list):
#         # Compute relative standard deviation for each coefficient
#         for col in ["C_D", "C_S", "C_L", "C_roll", "C_pitch", "C_yaw"]:

#             if col in ["C_S", "C_roll", "C_yaw"]:
#                 # filter the df, only take sideslip values that are positive > 0
#                 df = df[df["sideslip"] > 0]

#             mean_mean = np.abs(df[f"{col}_mean"].mean())
#             std_mean = df[f"{col}_std"].mean()
#             rel_std_mean = std_mean / mean_mean
#             table_data[Re][col] = f"{rel_std_mean:.2f}"

#         print(f"\nRe: {Re}")
#         # Compute mean SNR for each variable
#         for snr_col in [
#             "SNR_CF_X",
#             "SNR_CF_Y",
#             "SNR_CF_Z",
#             "SNR_CM_X",
#             "SNR_CM_Y",
#             "SNR_CM_Z",
#         ]:
#             if snr_col in df.columns:
#                 mean_snr = df[snr_col].mean()
#                 table_data[Re][snr_col] = f"{mean_snr:.2f}"

#     for Re in Reynolds_list:
#         print(f"\nRe: {Re}")
#         # print(table_data[Re])
#         for key, value in table_data[Re].items():
#             print(f"{key}: {value}")

#     return pd.DataFrame.from_dict(table_data, orient="index")


if __name__ == "__main__":
    main(project_dir)
