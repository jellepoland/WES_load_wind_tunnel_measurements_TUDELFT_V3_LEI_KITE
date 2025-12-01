from pathlib import Path
from plot_styling import set_plot_style
import matplotlib.pyplot as plt
from load_balance_analysis import plot_zigzag
from load_balance_analysis import plot_literature_comparison
from load_balance_analysis import plot_reynolds_variation
from load_balance_analysis import plot_uncertainty_boxplots
from load_balance_analysis import print_kite_dimensions
from load_balance_analysis import plot_and_print_sensor_drift
from load_balance_analysis import print_relative_standard_deviation
from load_balance_analysis import print_repeatability_uncertainty
from load_balance_analysis import print_min_max
from load_balance_analysis import print_kite_cg
from load_balance_analysis import plot_critical_trip_height
from load_balance_analysis import plot_frequency_psd
from load_balance_analysis import plot_2D_polar_breukels
from load_balance_analysis import plot_lvm_sample

from load_balance_analysis.functions_utils import *


def main():
    set_plot_style(is_for_pdf_tex=True)

    results_path = Path(project_dir) / "results"
    # results_path = Path(
    #     "/home/jellepoland/ownCloud/phd/latex_documents/WES24_KITE_WindTunnel/Images"
    # )

    ## plotting
    plot_zigzag.main(results_path, project_dir)
    plot_literature_comparison.main(results_path, project_dir)
    plot_reynolds_variation.main(results_path, project_dir)
    plot_critical_trip_height.main(results_path, project_dir)
    plot_frequency_psd.main(results_path, project_dir)
    plot_and_print_sensor_drift.main(results_path, project_dir)
    plot_lvm_sample.main(results_path, project_dir)

    # printing
    # print_kite_dimensions.main(project_dir)
    # print_relative_standard_deviation.main(project_dir)
    # print_repeatability_uncertainty.main(project_dir)
    # print_min_max.main(project_dir)
    # print_kite_cg.main(project_dir)

    print(f"\n--> New plots and tables generated, and saved inside: \n {results_path}")


if __name__ == "__main__":
    main()
