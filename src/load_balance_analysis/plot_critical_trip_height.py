import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.ticker import ScalarFormatter
from load_balance_analysis.functions_utils import (
    project_dir,
    save_latex_table,
    y_axis_labels,
)
from plot_styling import set_plot_style, plot_on_ax

# kcrit_save_file_path = Path(project_dir) / "results" / "kcrit_seam_height.png"
# Rek_save_file_path = Path(project_dir) / "results" / "rek_seam_height.png"


# # Constants
# rho = 1.225  # Air density in kg/m^3 (standard at sea level)
# visc_dyn = 1.8e-5  # Dynamic viscosity in Pa·s
# visc_kin = 1.5e-5  # Kinematic viscosity in m^2/s
# Cp = -1.4  # Pressure coefficient
# bl_f = 0.8  # Boundary layer factor


# # Function to calculate Rek
# def calculate_rek(Re, c, k):
#     """
#     Calculate the roughness Reynolds number (Rek) for given parameters.
#     """
#     # Calculate flow velocity from Reynolds number and chord length
#     v_inf = Re * visc_kin / c  # Flow velocity based on Reynolds number
#     # Calculate local velocity
#     v_local = bl_f * (v_inf * np.sqrt(1 - Cp))
#     # Calculate Rek
#     Rek = v_local * k / visc_dyn
#     return Rek


# # Input values
# c_values = [1.6, 5.0]  # Chord lengths in meters
# k_values = [0.0008, 0.001]  # Trip heights in meters
# Re_values = np.linspace(0.8e6, 5e6, 10)  # Range of Reynolds numbers

# # Plotting
# plt.figure(figsize=(8, 5))
# for c in c_values:
#     for k in k_values:
#         Rek_values = [calculate_rek(Re, c, k) for Re in Re_values]
#         label = f"c = {c} m, k = {k} m"
#         plt.plot(Re_values, Rek_values, label=label)

# # Add horizontal line for min Rek threshold
# plt.axhline(y=400, color="k", linestyle="--", label="Min Rek (400)")
# plt.axhline(y=600, color="b", linestyle="--", label="Min Rek (600)")

# # Formatting the plot
# ax = plt.gca()
# ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
# ax.ticklabel_format(style="sci", axis="x", scilimits=(6, 6))
# plt.xlabel("Reynolds Number (Re)")
# plt.ylabel("Roughness Reynolds Number (Rek)")
# plt.title(f"cp: {Cp}, boundary layer factor: {bl_f}")
# plt.grid()
# plt.legend()
# plt.savefig(Rek_save_file_path, dpi=300)
# plt.show()


# # Function to calculate kcrit based on Re, xtr, and c
# def calculate_kcrit(Re, xtr, c):
#     U = Re / c * visc_kin  # Flow velocity
#     x = xtr * c  # Transition location in meters
#     print(x)
#     Rx = U * x / visc_kin  # Local Reynolds number at trip location

#     if Rx > 2e5:
#         kcrit = 52 * x * Rx ** (-0.765)  # in m
#     else:
#         kcrit = 63.2 * x * Rx ** (-1.43) + 1.4 * x * Rx ** (-0.5)  # in m
#     return kcrit, Rx


# xtr_values = [0.02, 0.03]  # Transition locations (%c)

# # Generate a range of Reynolds numbers
# Re_values = np.linspace(1e5, 8e8, 10)  # From 1e6 to 1e8 for Reynolds numbers

# # Plot the results
# plt.figure(figsize=(8, 5))

# # Loop through both c and xtr values and plot kcrit
# for c in c_values:
#     for xtr in xtr_values:
#         kcrit_values = [calculate_kcrit(Re, xtr, c)[0] for Re in Re_values]
#         Rx = [calculate_kcrit(Re, xtr, c)[1] for Re in Re_values]
#         label = f"c = {c} m, xtr = {xtr*100:.0f}%"  # Label format: 'c = x m, xtr = y%'
#         plt.plot(Re_values, kcrit_values, label=label)

# # Set the x-axis to logarithmic and set ticks manually
# plt.xscale("log")  # Use logarithmic scale for x-axis
# xticks_pos = [1e6, 5e6, 1e7, 5e7, 1e8, 5e8]  # Ticks at specified positions
# xticks_labels = ["1e6", "5e6", "1e7", "5e7", "1e8", "5e8"]  # Custom labels
# plt.xticks(xticks_pos, xticks_labels)


# seam_height = 0.001  # [m]
# seam_height2 = 0.0008  # [m]
# plt.axhline(
#     y=seam_height, color="k", linestyle="--", label=f"Trip height = {seam_height} m"
# )
# plt.axhline(
#     y=seam_height2, color="b", linestyle="--", label=f"Trip height = {seam_height2} m"
# )


# # Labeling the axes and the title
# plt.xlabel("Re [-]", fontsize=12)
# plt.ylabel("Critical Trip Height (kcrit) [mm]", fontsize=12)
# plt.title("Critical Seam Height vs Reynolds Number", fontsize=14)

# # Display grid, legend, and plot
# plt.grid(True)
# plt.legend()

# # Save the figure (optional)
# plt.savefig(kcrit_save_file_path, dpi=300)

# # Show the plot
# plt.show()


#############################################
## Jelle Plot
#############################################


def kinematic_viscosity_air(T_deg=27, rho=1.19):
    """
    Calculate kinematic viscosity of air.

    Parameters:
        T (float): Temperature in Kelvin.
        p (float): Pressure in Pascals (default is standard atmospheric pressure).

    Returns:
        float: Kinematic viscosity in m^2/s.
    """
    # Constants
    T = T_deg + 273.15  # Temperature in Kelvin
    mu_0 = 1.716e-5  # Reference dynamic viscosity (Pa.s)
    T_0 = 273.15  # Reference temperature (K)
    C = 110.4  # Sutherland constant (K)
    R = 287.05  # Specific gas constant for air (J/(kg.K))

    # Dynamic viscosity (Sutherland's law)
    mu = mu_0 * ((T / T_0) ** 1.5) * (T_0 + C) / (T + C)

    # Kinematic viscosity
    nu = mu / rho
    return nu


def calculate_kcrit(Re, xtr, c, nu):
    U = Re / c * nu  # Flow velocity
    x = xtr * c  # Transition location in meters

    Rx = U * x / nu  # Local Reynolds number at trip location

    if Rx > 2e5:
        kcrit = 52 * x * Rx ** (-0.765)  # in m
    else:
        kcrit = 63.2 * x * Rx ** (-1.43) + 1.4 * x * Rx ** (-0.5)  # in m
    # print(
    #     f"Re:{Re/1e5:.0f}1e5, U: {U:.1f}m/s, Re_x = {Rx/1e5:.1f}1e5, kcrit: {kcrit*1e3:.1f}mm"
    # )
    return kcrit, Rx


# def main(results_path, project_dir):

#     # Kinematic viscosity of air at 27°C and 1.19 kg/m^3
#     nu = kinematic_viscosity_air(T_deg=27, rho=1.19)
#     print(f"nu: {nu:.2e} m^2/s")
#     c_values = [0.395]  # Chord lengths in meters
#     xtr_values = [0.05]  # Transition locations (%c)

#     Re_values = np.array([1.4, 2.8, 4.2, 5.6, 6.9]) * 1e5

#     # Plot the results
#     set_plot_style()
#     fig, ax = plt.subplots(figsize=(6.07, 3.57))

#     crit = "crit"
#     # Loop through both c and xtr values and plot kcrit
#     for c in c_values:
#         for xtr in xtr_values:
#             # print(f"\nxtr: {xtr}m, c: {c}m, transition location: {xtr * c:.2f}m")
#             kcrit_values = [calculate_kcrit(Re, xtr, c, nu)[0] for Re in Re_values]
#             Rx = [calculate_kcrit(Re, xtr, c, nu)[1] for Re in Re_values]
#             # label = f"c = {c} m, xtr = {xtr*100:.0f}%"
#             label = f"Minimal trip height"

#             # Plot using plot_on_ax
#             plot_on_ax(
#                 ax=ax,
#                 x=Re_values / 1e5,
#                 y=np.array(kcrit_values) * 1e3,
#                 label=label,
#                 marker="o",
#                 markersize=5,
#                 x_label=rf"Re $\times 10^5$ [-]",
#                 y_label=y_axis_labels["kcrit"],
#             )

#     trip_height = 0.2  # [mm]
#     ax.axhline(
#         y=trip_height,
#         color="red",
#         linestyle="--",
#         label=f"Set value",
#     )
#     ax.legend()
#     plt.tight_layout()
#     plt.savefig(Path(results_path) / "kcrit_seam_height.pdf")


def calculate_kcrit_rougnessRe(U, nu, Rek_crit):
    return Rek_crit * nu / U


def main(results_path, project_dir):
    # nu = kinematic_viscosity_air(T_deg=20, rho=1.19)
    nu = 1.56e-5
    print(f"nu: {nu:.2e} m^2/s")

    set_plot_style()
    fig, ax = plt.subplots(figsize=(6.2, 3.7))

    U_arr = np.array([5, 10, 15, 20, 25])  # Flow velocities in m/s
    Rek_crit_arr = np.array([200, 300])  # Roughness Reynolds number

    # Define different markers for each line
    markers = ["o", "x", "d", "s"]
    linestyles = ["-", "-."]

    print(10 * 0.396 / nu)
    U_arr = np.linspace(5, 25, 50)  # Flow velocities in m/s
    Re = U_arr * 0.396 / nu
    # plot for each U, and lines with different Rek_crit
    for i, Rek_crit in enumerate(Rek_crit_arr):
        k_crit_arr = Rek_crit * nu / U_arr  # Calculate critical roughness height
        k_crit_mm = k_crit_arr * 1000  # Convert to mm

        # Plot the line with specified marker but default color
        line = ax.plot(
            Re / 1e5,
            k_crit_mm,
            # marker=markers[i],
            linestyle=linestyles[i],
            label=r"$\textrm{Re}_{\textrm{k,crit}}$" + f" = {Rek_crit}",
            color="black",
        )

    # Set labels and title
    ax.set_xlabel(r"$\textrm{Re} \times 10^{-5}$ (-)")
    U_arr = np.array([5, 10, 15, 20, 25])  # Flow velocities in m/s
    re_arr = [1.3, 2.5, 3.8, 5, 6.1]
    ax.set_xticks(re_arr)
    ax.set_xlim([1.3, 6.1])
    ax.set_ylabel(rf"$k$ (mm)")

    # Add reference line for the trip height
    trip_height = 0.2  # [mm]
    ax.axhline(y=trip_height, color="red", linestyle="--", label=rf"$k=0.2$")

    ax.legend(ncols=1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(results_path) / "kcrit_seam_height.pdf")
