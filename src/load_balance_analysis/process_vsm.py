import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path
from load_balance_analysis.functions_utils import project_dir

from VSM.core.WingGeometry import Wing
from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.core.Solver import Solver
from VSM.plotting import generate_3D_polar_data
from VSM.plot_geometry_plotly import interactive_plot


def save_polar_data(
    angle_range,
    angle_type,
    angle_of_attack,
    name_appendix,
    body_aero,
    solver,
    solver_stall=None,
    vw=None,
):
    polar_data, reynolds_number = generate_3D_polar_data(
        solver=solver,
        body_aero=body_aero,
        angle_range=angle_range,
        angle_type=angle_type,
        angle_of_attack=angle_of_attack,
        side_slip=0,
        yaw_rate=0,
        Umag=vw,
    )

    print(f"\nReynolds number: {reynolds_number/1e5:.3f}e5\n")
    # Create dataframe and save to CSV
    if angle_type == "angle_of_attack":
        angle = "aoa"
        file_name = f"VSM_results_alpha_sweep_Rey_{(reynolds_number/1e5):.1f}"
    elif angle_type == "side_slip":
        angle = "beta"
        file_name = f"VSM_results_beta_sweep_Rey_{(reynolds_number/1e5):.1f}_alpha_{angle_of_attack*100:.0f}"
    else:
        raise ValueError("angle_type must be either 'angle_of_attack' or 'side_slip'")

    polar_dir = Path(project_dir) / "processed_data" / "polar_data"
    ### saving  ###
    path_to_csv = Path(polar_dir) / f"{file_name}{name_appendix}.csv"
    pd.DataFrame(
        {
            str(angle): polar_data[0],
            "CL": polar_data[1],
            "CD": polar_data[2],
            "CL/CD": np.array(polar_data[1]) / np.array(polar_data[2]),
            "CS": polar_data[3],
        }
    ).to_csv(path_to_csv, index=False)
    path_to_csv = Path(polar_dir) / f"{file_name}{name_appendix}_moment.csv"
    pd.DataFrame(
        {
            str(angle): polar_data[0],
            "CL": polar_data[1],
            "CD": polar_data[2],
            "CL/CD": np.array(polar_data[1]) / np.array(polar_data[2]),
            "CS": polar_data[3],
            "CMx": polar_data[4],
            "CMy": polar_data[5],
            "CMz": polar_data[6],
        }
    ).to_csv(path_to_csv, index=False)


def main():

    # breakpoint()

    te_point_full = np.array([1.443146003226444, 0, 3.754972573823276])
    x_displacement_from_te_full = -0.157 * 6.5
    z_displacement_from_te_full = -0.252 * 6.5

    ### FULL
    geom_scaling = 1.0
    x_displacement_from_te = x_displacement_from_te_full / geom_scaling
    z_displacement_from_te = z_displacement_from_te_full / geom_scaling
    ref_point_from_te_edge = np.array(
        [x_displacement_from_te, 0, z_displacement_from_te]
    )
    reference_point = te_point_full / geom_scaling + ref_point_from_te_edge
    print(f"FULL reference_point: {reference_point}")

    # ### SCALED
    # geom_scaling = 6.5
    # x_displacement_from_te = x_displacement_from_te_full / geom_scaling
    # z_displacement_from_te = z_displacement_from_te_full / geom_scaling
    # ref_point_from_te_edge = np.array(
    #     [x_displacement_from_te, 0, z_displacement_from_te]
    # )
    # reference_point = te_point_full / geom_scaling + ref_point_from_te_edge
    # print(f"SCALED reference_point: {reference_point}")

    # # TODO: pre 23-10-2025, using cg
    # # Reference Point
    # te_point_full_size_CAD = np.array([1.443146003226444, 0, 3.754972573823276])
    # te_point_full_size_CAD[2] += 7.25
    # geom_scaling = 1.0  # TODO: 13/02/2026 changed this to no scaling
    # x_displacement_from_te = (
    #     -0.157 * 6.5 / geom_scaling
    # )  # -0.172 # TODO: 13/02/2026 changed this to no scaling
    # z_displacement_from_te = (
    #     -0.252 * 6.5 / geom_scaling
    # )  # TODO: 13/02/2026 changed this to no scaling
    # ref_point_from_te_edge = np.array(
    #     [x_displacement_from_te, 0, z_displacement_from_te]
    # )
    # reference_point = te_point_full_size_CAD / geom_scaling + ref_point_from_te_edge
    # print(f"FULL reference_point: {reference_point}")

    # ### SCALED
    # te_point_full_size_CAD = np.array([1.443146003226444, 0, 3.754972573823276])
    # geom_scaling = 6.5
    # x_displacement_from_te = -0.157
    # z_displacement_from_te = -0.252
    # ref_point_from_te_edge = np.array(
    #     [x_displacement_from_te, 0, z_displacement_from_te]
    # )
    # reference_point = te_point_full_size_CAD / geom_scaling + ref_point_from_te_edge
    # print(f"SCALED reference_point: {reference_point}")

    # breakpoint()
    # TODO: corrected the ref point
    # 16/02/2026
    # te_ref_point_full_new = np.array([1.443146003226444, 0.0, 11.004972573823276])
    # reference_point = te_ref_point_full_new + ref_point_from_te_edge

    ## Settings
    vw_scaled = 18.52
    vw_full = 2.82
    vw = vw_full
    n_panels = 150
    spanwise_panel_distribution = "uniform"

    ### create body_aero
    geom_scaled_path_scaled = (
        Path(project_dir) / "data" / "vsm_input" / "wing_geometry_scaled.yaml"
    )
    geom_scaled_path_full = (
        Path(project_dir)
        / "data"
        / "vsm_input"
        / "2D_airfoils_polars_plots_BEST"
        / "aero_geometry_CFD_CAD_derived_z_min7_25.yaml"
    )
    body_aero = BodyAerodynamics.instantiate(
        n_panels=n_panels,
        file_path=geom_scaled_path_full,
        spanwise_panel_distribution=spanwise_panel_distribution,
    )
    solver = Solver(reference_point=reference_point)

    # Computing the Reynolds number
    rho = 1.16  # kg/m^3
    mu = 1.68e-5  # kg/(m*s) #for 28.7deg
    ref_length = (
        2.59 / geom_scaling
    )  # m, using the chord length at the reference point as the reference length for Reynolds number calculation
    kinematic_viscosity = mu / rho
    reynolds_number = (vw * ref_length) / kinematic_viscosity
    print(f"\nReynolds number: {reynolds_number/1e5:.3f}e5\n")

    ### alpha sweeps
    alphas_to_be_plotted = np.linspace(-12, 25, 38)
    save_polar_data(
        angle_range=alphas_to_be_plotted,
        angle_type="angle_of_attack",
        angle_of_attack=0,
        name_appendix="_corrected",
        body_aero=body_aero,
        solver=solver,
        solver_stall=None,
        vw=vw,
    )
    ### beta sweeps
    betas_to_be_plotted = np.linspace(0, 20, 21)
    save_polar_data(
        angle_range=betas_to_be_plotted,
        angle_type="side_slip",
        angle_of_attack=7.4,
        name_appendix="_corrected",
        body_aero=body_aero,
        solver=solver,
        solver_stall=None,
        vw=vw,
    )
    save_polar_data(
        angle_range=betas_to_be_plotted,
        angle_type="side_slip",
        angle_of_attack=12.5,
        name_appendix="_corrected",
        body_aero=body_aero,
        solver=solver,
        solver_stall=None,
        vw=vw,
    )


if __name__ == "__main__":
    main()
