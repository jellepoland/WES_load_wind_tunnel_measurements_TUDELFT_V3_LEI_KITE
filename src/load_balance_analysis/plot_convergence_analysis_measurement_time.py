import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from plot_styling import plot_on_ax, set_plot_style
from load_balance_analysis.functions_utils import project_dir, saving_pdf_and_pdf_tex


def block_averaged_statistics(data, n_blocks=5):
    """
    Divide data into N equal-length blocks and compute statistics for each block.

    Args:
        data (np.ndarray): 1D time series data
        n_blocks (int): Number of blocks to divide the data into

    Returns:
        dict: Dictionary containing block statistics
    """
    # Calculate block size
    block_size = len(data) // n_blocks

    # Trim data to fit exactly into blocks
    trimmed_data = data[: block_size * n_blocks]

    # Reshape data into blocks
    blocks = trimmed_data.reshape(n_blocks, block_size)

    # Compute statistics for each block
    block_stats = {"block_means": [], "block_stds": [], "block_indices": []}

    print(
        f"\nBlock-averaged statistics (N = {n_blocks} blocks, block size = {block_size}):"
    )
    print("-" * 60)

    for i in range(n_blocks):
        block_mean = np.mean(blocks[i])
        block_std = np.std(blocks[i], ddof=1)  # Sample standard deviation

        block_stats["block_means"].append(block_mean)
        block_stats["block_stds"].append(block_std)
        block_stats["block_indices"].append(i + 1)

        print(f"Block {i+1:2d}: Mean = {block_mean:8.4f}, Std = {block_std:8.4f}")

    print("-" * 60)

    return block_stats


def compute_running_average(data):
    """
    Compute the running average (cumulative mean) of the data.

    Args:
        data (np.ndarray): 1D time series data

    Returns:
        np.ndarray: Running average of the data
    """
    # Compute cumulative sum and divide by sample index
    cumsum = np.cumsum(data)
    sample_indices = np.arange(1, len(data) + 1)
    running_avg = cumsum / sample_indices

    return running_avg


def plot_block_statistics(block_stats, save_path):
    """
    Plot block-averaged statistics.

    Args:
        block_stats (dict): Dictionary containing block statistics
        save_path (Path): Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot block means
    plot_on_ax(
        ax1,
        block_stats["block_indices"],
        block_stats["block_means"],
        label="Block means",
        linestyle="-",
        marker="o",
        title="Block-averaged means",
    )
    ax1.set_xlabel("Block index (-)")
    ax1.set_ylabel("Mean value (-)")
    ax1.grid(True)

    # Plot block standard deviations
    plot_on_ax(
        ax2,
        block_stats["block_indices"],
        block_stats["block_stds"],
        label="Block standard deviations",
        linestyle="-",
        marker="s",
        title="Block-averaged standard deviations",
    )
    ax2.set_xlabel("Block index (-)")
    ax2.set_ylabel("Standard deviation (-)")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Block statistics plot saved to {save_path}")


def plot_running_average_analysis(data, running_avg, save_path, max_samples=None):
    """
    Plot the original time series and running average for comparison.

    Args:
        data (np.ndarray): Original 1D time series data
        running_avg (np.ndarray): Running average of the data
        save_path (Path): Path to save the plot
        max_samples (int, optional): Maximum number of samples to plot for clarity
    """
    # Create sample indices
    sample_indices = np.arange(len(data))

    # Limit plotting range if specified
    if max_samples is not None and len(data) > max_samples:
        plot_indices = sample_indices[:max_samples]
        plot_data = data[:max_samples]
        plot_running_avg = running_avg[:max_samples]
    else:
        plot_indices = sample_indices
        plot_data = data
        plot_running_avg = running_avg

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot original time series
    plot_on_ax(
        ax1,
        plot_indices,
        plot_data,
        label="Original signal",
        linestyle="-",
        alpha=0.7,
        title="Original time series",
    )
    ax1.set_ylabel("Signal amplitude (-)")
    ax1.grid(True)
    ax1.legend(loc="best")

    # Plot running average comparison
    plot_on_ax(
        ax2,
        plot_indices,
        plot_data,
        label="Original signal",
        linestyle="-",
        alpha=0.5,
        title="Running average analysis",
    )
    plot_on_ax(
        ax2,
        plot_indices,
        plot_running_avg,
        label="Running average",
        linestyle="-",
        linewidth=2,
        title="Running average analysis",
    )
    ax2.set_xlabel("Sample index (-)")
    ax2.set_ylabel("Signal amplitude (-)")
    ax2.grid(True)
    ax2.legend(loc="best")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Running average analysis plot saved to {save_path}")


def analyze_time_series_statistics(
    data, n_blocks=5, save_dir=None, max_plot_samples=None
):
    """
    Complete analysis of time series statistics including block averaging and running average.

    Args:
        data (np.ndarray): 1D time series data
        n_blocks (int): Number of blocks for block averaging
        save_dir (Path, optional): Directory to save plots
        max_plot_samples (int, optional): Maximum samples to plot for clarity

    Returns:
        dict: Dictionary containing all computed statistics
    """
    print(f"Analyzing time series with {len(data)} samples")

    # Block-averaged statistics
    block_stats = block_averaged_statistics(data, n_blocks)

    # Running average analysis
    running_avg = compute_running_average(data)

    # Print overall statistics
    print(f"\nOverall statistics:")
    print(f"Overall mean: {np.mean(data):.4f}")
    print(f"Overall std:  {np.std(data, ddof=1):.4f}")
    print(f"Final running average: {running_avg[-1]:.4f}")

    # Generate plots if save directory is provided
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Plot block statistics
        block_stats_path = save_dir / "block_statistics.pdf"
        plot_block_statistics(block_stats, block_stats_path)

        # Plot running average analysis
        running_avg_path = save_dir / "running_average_analysis.pdf"
        plot_running_average_analysis(
            data, running_avg, running_avg_path, max_plot_samples
        )

    # Return all results
    results = {
        "block_stats": block_stats,
        "running_average": running_avg,
        "overall_mean": np.mean(data),
        "overall_std": np.std(data, ddof=1),
    }

    return results


def main(results_dir: Path, project_dir: Path) -> None:
    """
    Main function to run time series analysis with synthetic data.

    Args:
        results_dir (Path): Directory to save results
        project_dir (Path): Project directory
    """
    set_plot_style(is_for_pdf_tex=True)

    # Generate synthetic time series data (similar to force measurements)
    np.random.seed(42)  # For reproducible results
    n_samples = 10000
    time = np.linspace(0, 10, n_samples)  # 10 seconds of data

    # Create a signal with trend, oscillation, and noise (similar to wind tunnel data)
    trend = 0.1 * time  # Linear trend
    oscillation = 2.0 * np.sin(2 * np.pi * 0.5 * time)  # 0.5 Hz oscillation
    noise = 0.5 * np.random.randn(n_samples)  # Random noise

    data = trend + oscillation + noise

    # Analyze the synthetic data
    results = analyze_time_series_statistics(
        data,
        n_blocks=5,
        save_dir=results_dir,
        max_plot_samples=2000,  # Plot only first 2000 samples for clarity
    )

    print(f"\n--> Time series analysis complete! Plots saved to: {results_dir}")

    return results


if __name__ == "__main__":
    results_dir = Path(project_dir) / "results"
    main(results_dir, project_dir)
