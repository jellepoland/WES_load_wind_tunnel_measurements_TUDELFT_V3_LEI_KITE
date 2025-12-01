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
    vw=1.05,
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


def running_vsm_to_generate_csv_data(
    project_dir: str,
    vw: float,
    geom_scaling,
    is_with_corrected_polar,
    reference_point,
    n_panels,
    spanwise_panel_distribution,
) -> None:
    if is_with_corrected_polar:
        print("Running VSM with corrected polar")
        name_appendix = "_corrected"
    else:
        print("Running VSM with breukels polar")
        name_appendix = "_breukels"

    # vsm_input_path = Path(project_dir) / "data" / "vsm_input"

    # ## scaling down geometry
    # geom_path = Path(vsm_input_path) / "wing_geometry_from_CAD_orderded_tip_to_mid.csv"
    # geom_scaled_path = (
    #     Path(vsm_input_path) / "wing_geometry_from_CAD_orderded_tip_to_mid_scaled.csv"
    # )
    # df = pd.read_csv(geom_path, delimiter=",")  # , skiprows=1)
    # df["LE_x"] = df["LE_x"].values / geom_scaling
    # df["LE_y"] = df["LE_y"].values / geom_scaling
    # df["LE_z"] = df["LE_z"].values / geom_scaling
    # df["TE_x"] = df["TE_x"].values / geom_scaling
    # df["TE_y"] = df["TE_y"].values / geom_scaling
    # df["TE_z"] = df["TE_z"].values / geom_scaling
    # df.to_csv(geom_scaled_path, index=False)

    ### create body_aero
    vsm_input_path = Path(project_dir) / "data" / "vsm_input"
    geom_scaled_path = Path(vsm_input_path) / "wing_geometry_scaled.yaml"
    body_aero = BodyAerodynamics.instantiate(
        n_panels=n_panels,
        file_path=geom_scaled_path,
        spanwise_panel_distribution=spanwise_panel_distribution,
    )
    solver = Solver(reference_point=reference_point)

    ### Plotting reference point at mid-span plane
    # plt.figure()
    # df_row_18 = df.iloc[18]
    # LE = df_row_18[["LE_x", "LE_y", "LE_z"]].values
    # TE = df_row_18[["TE_x", "TE_y", "TE_z"]].values
    # plt.plot(LE[0], LE[2], "ro", label="LE")
    # plt.plot(TE[0], TE[2], "bo", label="TE")
    # plt.plot((TE[0] - 0.395), TE[2], "bx", label="LE (approximate)")
    # plt.plot(reference_point[0], reference_point[2], "go", label="Reference point")
    # plt.legend()
    # plt.axis("equal")
    # plt.grid()
    # plt.show()
    # plt.close()

    # ### INTERACTIVE PLOT
    # interactive_plot(
    #     body_aero,
    #     vel=3.15,
    #     angle_of_attack=6.75,
    #     side_slip=0,
    #     yaw_rate=0,
    #     is_with_aerodynamic_details=True,
    # )
    # breakpoint()

    ### alpha sweeps
    alphas_to_be_plotted = np.linspace(-12, 25, 38)
    save_polar_data(
        angle_range=alphas_to_be_plotted,
        angle_type="angle_of_attack",
        angle_of_attack=0,
        name_appendix=name_appendix,
        body_aero=body_aero,
        solver=solver,
        solver_stall=None,
        vw=vw,
    )
    ### beta sweeps
    betas_to_be_plotted = np.linspace(0, 20, 20)
    save_polar_data(
        angle_range=betas_to_be_plotted,
        angle_type="side_slip",
        angle_of_attack=7.4,
        name_appendix=name_appendix,
        body_aero=body_aero,
        solver=solver,
        solver_stall=None,
        vw=vw,
    )
    save_polar_data(
        angle_range=betas_to_be_plotted,
        angle_type="side_slip",
        angle_of_attack=12.5,
        name_appendix=name_appendix,
        body_aero=body_aero,
        solver=solver,
        solver_stall=None,
        vw=vw,
    )
    return


def main():

    ## scaled down geometry
    # vw = 2.82
    # geom_scaling = 6.5
    # n_panels = 40

    # # ## scaled down velocity
    # # vw = 3.05
    # # geom_scaling = 1.0

    # ## Computing the reference point, to be equal as used for calc. the wind tunnel data Moments
    # x_displacement_from_te = -0.157  # -0.172
    # z_displacement_from_te = -0.252
    # te_point_full_size_surfplan = np.array([1.472144, 0, 3.696209])
    # te_point_full_size_CAD = np.array(
    #     [1.443146003226444, 8.28776104036884e-10, 3.754972573823276]
    # )
    # te_point_scaled = te_point_full_size_CAD / geom_scaling
    # ## height was off even tho chord and span are matching perfectly...
    # height_correction_factor = 1.0
    # te_point_scaled[2] = te_point_scaled[2] * height_correction_factor
    # reference_point = te_point_scaled + np.array(
    #     [x_displacement_from_te, 0, z_displacement_from_te]
    # )
    # # breakpoint()

    # TODO: pre 23-10-2025, using cg
    # Reference Point
    te_point_full_size_CAD = np.array(
        [1.443146003226444, 8.28776104036884e-10, 3.754972573823276]
    )
    geom_scaling = 6.5
    x_displacement_from_te = -0.157
    z_displacement_from_te = -0.252
    ref_point_from_te_edge = np.array(
        [x_displacement_from_te, 0, z_displacement_from_te]
    )
    reference_point = te_point_full_size_CAD / geom_scaling + ref_point_from_te_edge
    print(f"reference_point: {reference_point}")
    # breakpoint()

    ##TODO: post 23-10-2025, using tow-point
    # geom_scaling = 6.5
    # reference_point = np.array([0, 0, -7.5]) / geom_scaling  # tow-point

    ## Settings
    vw = 18.5  # 2.82
    n_panels = 150
    spanwise_panel_distribution = "uniform"

    running_vsm_to_generate_csv_data(
        project_dir,
        vw=vw,
        geom_scaling=geom_scaling,
        is_with_corrected_polar=True,
        reference_point=reference_point,
        n_panels=n_panels,
        spanwise_panel_distribution=spanwise_panel_distribution,
    )


if __name__ == "__main__":
    main()

    ##TODO: below is a test to verify that the correct projected_area is computed
    # ### create body_aero
    # vsm_input_path = Path(project_dir) / "data" / "vsm_input"
    # geom_scaled_path = Path(vsm_input_path) / "wing_geometry_scaled.yaml"
    # body_aero = BodyAerodynamics.instantiate(
    #     n_panels=50,
    #     file_path=geom_scaled_path,
    #     spanwise_panel_distribution="uniform",
    # )
    # # set va
    # vw = 18.5
    # alpha = 7.4
    # body_aero.va_initialize(
    #     Umag=vw,
    #     angle_of_attack=alpha,
    #     side_slip=0,
    # )
    # # set solver
    # solver = Solver(reference_point=[0, 0, 0])
    # # solve
    # results = solver.solve(body_aero)
    # # print projected_area
    # print(
    #     f"projected_area: {results['projected_area']}, true-size: {results['projected_area']*6.5**2}"
    # )
