import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import welch, find_peaks
from load_balance_analysis.functions_utils import project_dir
import matplotlib.pyplot as plt
from plot_styling import plot_on_ax, set_plot_style
import json


# Function to read LVM files
def read_lvm(filename):
    df = pd.read_csv(filename, skiprows=21, delimiter="\t", engine="python")
    df.drop(
        columns=df.columns[-1], inplace=True
    )  # Assuming the last column is to be dropped
    df.columns = ["time", "F_X", "F_Y", "F_Z", "M_X", "M_Y", "M_Z"]
    df["Filename"] = os.path.basename(filename).replace("_unsteady.lvm", "")
    num_samples = len(df) // 19800
    df["sample index"] = sum([[i] * 19800 for i in range(num_samples)], [])
    selected_rows = 19800 * 17  # Limit processing to the first 17 samples
    df = df.iloc[:selected_rows]
    return df


# Function to analyze frequency using Welch's method
def analyze_frequency(data, dt):
    # f, Pxx = welch(
    #     data,
    #     fs=1 / dt,  # Sampling frequency
    #     nperseg=2048,  # Increase segment size for better resolution
    #     noverlap=1024,  # 50% overlap for smoother estimate
    #     window="hann",
    # )  # Hann window reduces spectral leakage
    from scipy.fftpack import fft
    from scipy.signal import periodogram

    f, Pxx = periodogram(data, fs=1 / dt)
    return f, Pxx


def normalize_psd(Pxx):
    """
    Normalize Power Spectral Density (PSD) values to [0, 1] range.

    Args:
        Pxx (np.ndarray): Original PSD values

    Returns:
        np.ndarray: Normalized PSD values
    """
    return (Pxx - Pxx.min()) / (Pxx.max() - Pxx.min())


def plot_frequency_peaks_multi_vw(all_data, save_path, x_max):
    """
    Plot frequency peaks for multiple wind speeds on the same plot.

    Args:
        all_data (dict): Dictionary with wind speeds as keys and frequency/PSD data
        save_path (Path): Path to save the plot
    """
    channels = ["F_X", "F_Y", "F_Z", "M_X", "M_Y", "M_Z"]
    channel_labels = [
        r"$F_{\mathrm{x}}$",
        r"$F_{\mathrm{y}}$",
        r"$F_{\mathrm{z}}$",
        r"$M_{\mathrm{x}}$",
        r"$M_{\mathrm{y}}$",
        r"$M_{\mathrm{z}}$",
    ]
    wind_speeds = list(all_data.keys())
    colors = ["red", "blue", "green", "orange", "purple"]

    # Create a 3x2 grid for the plots
    fig, axs = plt.subplots(2, 3, figsize=(15, 5))
    axs = axs.flatten()

    for i, channel in enumerate(channels):
        for vw, color in zip(wind_speeds, colors):
            # if int(vw) == 25:
            # continue
            # Extract frequency and normalized PSD for the current channel and wind speed
            f = np.array(all_data[vw][channel]["frequency"])
            Pxx_normalized = np.array(all_data[vw][channel]["psd_normalized"])

            # Ensure both arrays are the same length
            assert len(f) == len(
                Pxx_normalized
            ), "Frequency and PSD arrays must be the same length"

            # Find peaks using numpy arrays
            peaks, _ = find_peaks(Pxx_normalized, height=0)

            # Plot on each axis
            plot_on_ax(
                axs[i],
                f,
                Pxx_normalized,
                label=f"{vw}",
                linestyle="-",
                # color=color,
                title=channel_labels[i],
            )

            # axs[i].plot(f, Pxx_normalized, label=f"{vw}", linestyle="-")

            # Safely plot peaks
            # if len(peaks) > 0:
            # axs[i].plot(f[peaks], Pxx_normalized[peaks], ".", markersize=10)

            # axs[i].set_title(rf"{channel_labels[i]}")
            axs[i].set_xlim([0, x_max])  # Limit to 0-1000 Hz
            # axs[i].set_ylim([0, 1.05])  # Limit to 0-1 PSD
            if i >= 3:
                is_x_label = True
            else:
                is_x_label = False
            axs[i].set_xlabel("Hz (-)" if is_x_label else "")
            axs[i].tick_params(labelbottom=is_x_label)

            if i in [0, 3]:
                is_y_label = True
            else:
                is_y_label = False
            axs[i].set_ylabel("Normalized PSD (-)" if is_y_label else "")
            axs[i].tick_params(labelleft=is_y_label)
            # axs[i].vlines(5, 0, 1.05, color="red", linestyle="--")
            axs[i].grid(True)

        if i == 0:
            axs[i].legend(loc="best")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def collect_wind_speed_data(project_dir, target_wind_speeds=None):
    """
    Collect and bundle frequency analysis data for specified wind speeds across all AoA folders.

    Args:
        project_dir (str): Project directory
        target_wind_speeds (list, optional): List of wind speeds to analyze.
                                             Defaults to [5, 10, 15, 20, 25].

    Returns:
        dict: Dictionary with wind speeds as keys and frequency/PSD data
    """
    # Default wind speeds if not provided
    if target_wind_speeds is None:
        target_wind_speeds = [5, 10, 15, 20, 25]

    # Base paths
    normal_folder_path = Path(project_dir) / "data" / "normal"

    # Dictionary to store bundled data for each wind speed
    wind_speed_data = {}

    # Channels to analyze
    channels = ["F_X", "F_Y", "F_Z", "M_X", "M_Y", "M_Z"]

    # Iterate over all AoA folders
    for aoa_folder in os.listdir(normal_folder_path):
        aoa_folder_path = os.path.join(normal_folder_path, aoa_folder)
        if os.path.isdir(aoa_folder_path) and aoa_folder.startswith("aoa_"):
            # Iterate over each .lvm file in the AoA folder
            for lvm_file in os.listdir(aoa_folder_path):
                if lvm_file.endswith(".lvm") and not lvm_file.endswith(
                    "00_unsteady.lvm"
                ):
                    # Extract wind speed from the filename
                    lvm_vw = int(lvm_file.split("_vw_")[1].split("_")[0])

                    # Process only the target wind speeds
                    if lvm_vw in target_wind_speeds:
                        lvm_file_path = os.path.join(aoa_folder_path, lvm_file)
                        df = read_lvm(lvm_file_path)

                        # Ensure wind speed key exists
                        if lvm_vw not in wind_speed_data:
                            wind_speed_data[lvm_vw] = {}

                        # Compute frequency analysis for each channel
                        for channel in channels:
                            data = df[channel].values
                            dt = df["time"].diff().mean()

                            # Perform frequency analysis
                            f, Pxx = analyze_frequency(data, dt)

                            # Normalize PSD
                            Pxx_normalized = normalize_psd(Pxx)

                            # Store frequency and normalized PSD
                            wind_speed_data[lvm_vw][channel] = {
                                "frequency": f.tolist(),
                                "psd_normalized": Pxx_normalized.tolist(),
                            }

    return wind_speed_data


def save_wind_speed_data(wind_speed_data, output_path):
    """
    Save wind speed frequency analysis data to a JSON file.

    Args:
        wind_speed_data (dict): Dictionary of wind speed frequency data
        output_path (Path or str): Path to save the data
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(wind_speed_data, f)

    print(f"Wind speed frequency analysis data saved to {output_path}")


def load_wind_speed_data(input_path):
    """
    Load wind speed frequency analysis data from a file.

    Args:
        input_path (Path or str): Path to load the data from

    Returns:
        dict: Loaded wind speed frequency analysis data
    """
    # Load from JSON
    with open(input_path, "r") as f:
        loaded_data = json.load(f)

    return loaded_data


def plot_wind_speed_data(wind_speed_data, save_path, x_max):
    """
    Plot frequency peaks for wind speed data.

    Args:
        wind_speed_data (dict): Dictionary with wind speeds as keys and frequency/PSD data
        save_path (Path): Path to save the plot
    """
    # Generate plot if data is available
    if wind_speed_data:
        # Generate and save the plot
        plot_frequency_peaks_multi_vw(
            all_data=wind_speed_data,
            save_path=save_path,
            x_max=x_max,
        )

        print("Plotted frequency analysis for multiple wind speeds.")
    else:
        print("No data available for plotting.")


def plot_time_series_aoa_vw(aoa_folder, vw, project_dir, save_path, x_max):
    """
    Plot the time series for a single AoA and wind speed case.

    Args:
        aoa_folder (str): Name of the AoA folder (e.g., "aoa_5").
        vw (int): Wind speed to analyze.
        project_dir (str): Project directory containing the data.
        save_path (Path): Path to save the generated plot.
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    import os

    # Base path to the data
    normal_folder_path = Path(project_dir) / "data" / "normal" / aoa_folder

    # Find the .lvm file for the specified wind speed
    lvm_file = None
    for file in os.listdir(normal_folder_path):
        if file.endswith(".lvm") and not file.endswith("00_unsteady.lvm"):
            if f"_vw_{vw}_" in file:
                lvm_file = os.path.join(normal_folder_path, file)
                break

    # Check if the file was found
    if lvm_file is None:
        raise FileNotFoundError(f"No .lvm file found for AoA {aoa_folder} and vw {vw}")

    # Read the data using the provided read_lvm function
    df = read_lvm(lvm_file)

    # Channels to plot
    channels = ["F_X", "F_Y", "F_Z", "M_X", "M_Y", "M_Z"]
    channel_labels = [
        r"$F_{\mathrm{x}}$ (N)",
        r"$F_{\mathrm{y}}$ (N)",
        r"$F_{\mathrm{z}}$ (N)",
        r"$M_{\mathrm{x}}$ (Nm)",
        r"$M_{\mathrm{y}}$ (Nm)",
        r"$M_{\mathrm{z}}$ (Nm)",
    ]

    # Time vector
    time = df["time"].values

    # Create the plot
    fig, axs = plt.subplots(2, 3, figsize=(15, 6))
    axs = axs.flatten()

    for i, (channel, label) in enumerate(zip(channels, channel_labels)):
        data = df[channel].values

        # Plot the time series
        axs[i].plot(time, data, label=label, color="black")
        axs[i].set_title(label)
        axs[i].set_xlabel("Time (s)" if i >= 3 else "")
        axs[i].set_ylabel("Force/Moment (N/Nm)" if i % 3 == 0 else "")
        axs[i].set_xlim([0, x_max])  # Limit to 0-1000 Hz
        # axs[i].set_ylim([0, 1.05])  # Limit to 0-1 PSD
        if i >= 3:
            is_x_label = True
        else:
            is_x_label = False
        axs[i].set_xlabel("Time (s)" if is_x_label else "")
        axs[i].tick_params(labelbottom=is_x_label)

        if i == 0:
            y_label = "Force (N)"
            is_y_label = True
        elif i == 3:
            y_label = "Moment (Nm)"
            is_y_label = True
        else:
            is_y_label = False

        axs[i].set_ylabel(y_label if is_y_label else "")
        axs[i].tick_params(labelleft=is_y_label)
        # axs[i].vlines(5, 0, 1.05, color="red", linestyle="--")
        axs[i].grid(True)

    # Adjust layout
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Time series plot saved to {save_path}")


def main(results_dir, project_dir):
    set_plot_style()
    # Path for saving/loading data
    data_output_path = Path(project_dir) / "processed_data" / "frequency_psd_data.json"

    ## Collect and save wind speed data
    # wind_speed_data = collect_wind_speed_data(project_dir)
    # save_wind_speed_data(wind_speed_data, data_output_path)

    ## Plot wind speed data
    wind_speed_data = load_wind_speed_data(data_output_path)

    x_max = 10  # Maximum frequency to plot
    plot_save_path = (
        Path(results_dir)
        / f"freq_plot_multi_vw_all_aoa_bundled_{x_max}Hz_periodogram.pdf"
    )
    plot_wind_speed_data(wind_speed_data, plot_save_path, x_max)
    x_max = 100  # Maximum frequency to plot
    plot_save_path = (
        Path(results_dir)
        / f"freq_plot_multi_vw_all_aoa_bundled_{x_max}Hz_periodogram.pdf"
    )
    plot_wind_speed_data(wind_speed_data, plot_save_path, x_max)

    ####
    aoa_folder = "aoa_21"  # Replace with your AoA folder
    vw = 25  # Wind speed to analyze
    save_path = (
        Path(results_dir) / f"time_series_plot_vw{vw}_{aoa_folder}.pdf"
    )  # Replace with the desired save path

    # Generate the plot
    plot_time_series_aoa_vw(aoa_folder, vw, project_dir, save_path, x_max=1)
