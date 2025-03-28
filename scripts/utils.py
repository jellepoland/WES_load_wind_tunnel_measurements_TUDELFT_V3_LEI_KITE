from pathlib import Path
import matplotlib as mpl
import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.font_manager as fm
import numpy as np
import scipy.stats as stats


def saving_pdf_and_pdf_tex(results_dir: str, filename: str):
    plt.savefig(Path(results_dir) / f"{filename}.pdf")
    plt.close()


x_axis_labels = {
    "alpha": r"$\alpha$ ($^\circ$)",
    "beta": r"$\beta$ ($^\circ$)",
    "Re": r"Re $\times 10^5$ (-)",
}

y_axis_labels = {
    "CL": r"$C_{\mathrm{L}}$ (-)",
    "CD": r"$C_{\mathrm{D}}$ (-)",
    "CS": r"$C_{\mathrm{S}}$ (-)",
    "CMx": r"$C_{\mathrm{M,x}}$ (-)",
    "CMy": r"$C_{\mathrm{M,y}}$ (-)",
    "CMz": r"$C_{\mathrm{M,z}}$ (-)",
    "L/D": r"$L/D$ (-)",
    "kcrit": r"$k_{\mathrm{crit}}$ (mm)",
}

project_dir = Path(__file__).resolve().parent.parent


## Confidence Interval
def calculate_confidence_interval(data, alpha=0.01):

    # print(f"data: {data}")
    mean = data.mean()
    sem = stats.sem(data)
    critical_value = stats.t.ppf(1 - alpha / 2, df=len(data) - 1)
    conf_interval = critical_value * sem
    # print(f"mean: {mean}, conf_interval: {conf_interval}")
    return conf_interval


## Confidence Interval using BLOCK BOOTSTRAP
def block_bootstrap_confidence_interval(
    data, block_size=500, n_bootstrap=1000, alpha=0.01
):
    """
    Calculate the confidence interval using block bootstrapping to handle autocorrelation.

    Args:
        data (np.ndarray): Time series data.
        block_size (int): The size of the blocks to preserve autocorrelation.
        n_bootstrap (int): Number of bootstrap resamples to generate.
        alpha (float): Significance level (default: 0.05 for a 95% confidence interval).

    Returns:
        mean (float): The mean of the original data.
        conf_interval (tuple): The confidence interval (lower bound, upper bound).
    """
    data = np.array(data)
    n_data = len(data)

    # Step 1: Split data into blocks
    num_blocks = n_data // block_size
    blocks = [
        data[i : i + block_size] for i in range(0, num_blocks * block_size, block_size)
    ]

    # Step 2: Perform bootstrap resampling on blocks
    bootstrap_means = []
    for _ in range(n_bootstrap):
        # Resample blocks with replacement
        bootstrapped_data = np.concatenate(
            [blocks[i] for i in np.random.randint(0, len(blocks), num_blocks)]
        )
        bootstrap_means.append(np.mean(bootstrapped_data))

    # Step 3: Compute the mean of the original data
    mean = np.mean(data)

    # Step 4: Calculate the confidence interval from bootstrapped means
    lower_bound = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper_bound = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    confidence_interval = np.abs(np.abs(lower_bound) - np.abs(upper_bound))
    return confidence_interval


## Confidence Interval using HAC/Newey-West
def hac_newey_west_confidence_interval(data, alpha=0.01, max_lag=None):
    """
    Calculate the confidence interval using the HAC/Newey-West method to handle autocorrelation and heteroskedasticity.

    Args:
        data (np.ndarray): Time series data from wind tunnel aerodynamic experiment.
        alpha (float): Significance level (default: 0.01 for a 99% confidence interval).
        max_lag (int): Maximum lag to consider for autocorrelation. If None, it's set to n^(1/4) where n is the sample size.

    Returns:
        mean (float): The mean of the original data.
        conf_interval (float): The confidence interval (half-width).
    """
    data = np.array(data)
    n = len(data)

    # Calculate the mean
    mean = np.mean(data)

    # Calculate the variance
    centered_data = data - mean
    variance = np.mean(centered_data**2)

    # Set max_lag if not provided
    if max_lag is None:
        max_lag = int(n ** (1 / 4))

    # Calculate autocovariances up to max_lag
    auto_cov = np.array(
        [np.mean(centered_data[i:] * centered_data[:-i]) for i in range(1, max_lag + 1)]
    )

    # Calculate Newey-West weights
    weights = 1 - np.arange(1, max_lag + 1) / (max_lag + 1)

    # Calculate the HAC standard error
    hac_variance = variance + 2 * np.sum(weights * auto_cov)
    hac_se = np.sqrt(hac_variance / n)

    # Calculate the confidence interval
    t_value = stats.t.ppf(1 - alpha / 2, df=n - 1)
    conf_interval = t_value * hac_se

    return conf_interval


def reduce_df_by_parameter_mean_and_std(
    df: pd.DataFrame, parameter: str
) -> pd.DataFrame:
    """
    Reduces a dataframe to unique values of a parameter, averaging specified columns
    and adding standard deviations for coefficients.

    Parameters:
    df (pandas.DataFrame): The input dataframe
    parameter (str): Either 'aoa_kite' or 'sideslip'

    Returns:
    pandas.DataFrame: Reduced dataframe with averages and coefficient standard deviations
    """
    # All columns to average
    columns_to_average = [
        "C_L",
        "C_S",
        "C_D",
        "C_roll",
        "C_pitch",
        "C_yaw",
        # "F_X_raw",
        # "F_Y_raw",
        # "F_Z_raw",
        # "M_X_raw",
        # "M_Y_raw",
        # "M_Z_raw",
        "Rey",
        "vw",
    ]

    if parameter == "aoa_kite":
        columns_to_average += ["sideslip"]
    elif parameter == "sideslip":
        columns_to_average += ["aoa_kite"]
    else:
        raise ValueError("Invalid parameter")

    # Calculate means
    mean_df = df.groupby(parameter)[columns_to_average].mean()

    # Coefficient columns that also need standard deviation
    coef_columns = ["C_L", "C_S", "C_D", "C_roll", "C_pitch", "C_yaw"]

    # Calculate & Rename standard deviations for coefficients
    std_df = df.groupby(parameter)[coef_columns].std()
    std_df.columns = [f"{col}_std" for col in std_df.columns]

    # Calculate & rename CI
    ci_df = df.groupby(parameter)[coef_columns].apply(calculate_confidence_interval)
    ci_df = pd.DataFrame(ci_df.to_list(), columns=[f"{col}_CI" for col in coef_columns])

    # # Calculate & rename CI using block bootstrap
    # ci_block_df = df.groupby(parameter)[coef_columns].apply(
    #     block_bootstrap_confidence_interval
    # )
    # ci_block_df = pd.DataFrame(
    #     ci_block_df.to_list(), columns=[f"{col}_CI_block" for col in coef_columns]
    # )

    # # Calculate & rename CI using HAC/Newey-West
    # ci_hac_df = df.groupby(parameter)[coef_columns].apply(
    #     hac_newey_west_confidence_interval
    # )
    # ci_hac_df = pd.DataFrame(
    #     ci_hac_df.to_list(), columns=[f"{col}_CI_hac" for col in coef_columns]
    # )

    # Combine dataframes
    # result_df = pd.concat(
    #     [mean_df, std_df, ci_df, ci_block_df, ci_hac_df], axis=1
    # ).reset_index()
    # ci_df = ci_df.reset_index(drop=True)
    # result_df = pd.concat([mean_df, std_df, ci_df], axis=1).reset_index()
    result_df = pd.concat([mean_df, std_df], axis=1).reset_index()

    # Round the velocities to 0 decimal places
    result_df["vw"] = result_df["vw"].round(0)

    return result_df


if __name__ == "__main__":
    print(f"project_dir: {project_dir}")
    print(f"\nlabel_x:")
    for label_x, label_x_item in zip(x_axis_labels, x_axis_labels.items()):
        print(f"{label_x}, {label_x_item}")
    print(f"\nlabel_y:")
    for label_y, label_y_item in zip(y_axis_labels, y_axis_labels.items()):
        print(f"{label_y}, {label_y_item}")
