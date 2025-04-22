import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path
from load_balance_analysis.functions_utils import project_dir

from VSM.WingGeometry import Wing
from VSM.BodyAerodynamics import BodyAerodynamics
from VSM.Solver import Solver
from VSM.plotting import generate_3D_polar_data
from VSM.interactive import interactive_plot


def create_body_aero(
    file_path,
    n_panels,
    spanwise_panel_distribution,
    is_with_corrected_polar=False,
    path_polar_data_dir="",
    geom_scaling=1.0,
):

    df = pd.read_csv(file_path, delimiter=",")  # , skiprows=1)
    LE_x_array = df["LE_x"].values / geom_scaling
    LE_y_array = df["LE_y"].values / geom_scaling
    LE_z_array = df["LE_z"].values / geom_scaling
    TE_x_array = df["TE_x"].values / geom_scaling
    TE_y_array = df["TE_y"].values / geom_scaling
    TE_z_array = df["TE_z"].values / geom_scaling
    d_tube_array = df["d_tube"].values
    camber_array = df["camber"].values

    ## populating this list
    rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_19ribs = []

    for i in range(len(LE_x_array)):
        LE = np.array([LE_x_array[i], LE_y_array[i], LE_z_array[i]])
        TE = np.array([TE_x_array[i], TE_y_array[i], TE_z_array[i]])
        rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_19ribs.append(
            [LE, TE, ["lei_airfoil_breukels", [d_tube_array[i], camber_array[i]]]]
        )
    CAD_wing = Wing(n_panels, spanwise_panel_distribution)

    for i, CAD_rib_i in enumerate(
        rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_19ribs
    ):
        CAD_rib_i_0 = CAD_rib_i[0]
        CAD_rib_i_1 = CAD_rib_i[1]

        if is_with_corrected_polar:
            ### using corrected polar
            df_polar_data = pd.read_csv(
                Path(path_polar_data_dir) / f"corrected_polar_{i}.csv"
            )
            alpha = df_polar_data["alpha"].values
            cl = df_polar_data["cl"].values
            cd = df_polar_data["cd"].values
            cm = df_polar_data["cm"].values
            polar_data = ["polar_data", np.array([alpha, cl, cd, cm])]
            CAD_wing.add_section(CAD_rib_i_0, CAD_rib_i_1, polar_data)
        else:
            ### using breukels
            CAD_wing.add_section(CAD_rib_i_0, CAD_rib_i_1, CAD_rib_i[2])

    body_aero = BodyAerodynamics([CAD_wing])

    return body_aero


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
    # polar_data_stall, _ = generate_polar_data(
    #     solver=VSM_with_stall_correction,
    #     body_aero=body_aero,
    #     angle_range=angle_range,
    #     angle_type=angle_type,
    #     angle_of_attack=angle_of_attack,
    #     side_slip=0,
    #     yaw_rate=0,
    #     Umag=vw,
    # )
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
    ### save forces ###
    # no_stall
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
    # # stall
    # path_to_csv = Path(polar_dir) / f"{file_name}{name_appendix}_stall.csv"
    # pd.DataFrame(
    #     {
    #         str(angle): polar_data[0],
    #         "CL": polar_data_stall[1],
    #         "CD": polar_data_stall[2],
    #         "CL/CD": np.array(polar_data_stall[1]) / np.array(polar_data_stall[2]),
    #         "CS": polar_data_stall[3],
    #     }
    # ).to_csv(path_to_csv, index=False)
    ### save moments ###
    # no_stall
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
    # # stall
    # path_to_csv = Path(polar_dir) / f"{file_name}{name_appendix}_stall_moment.csv"
    # pd.DataFrame(
    #     {
    #         str(angle): polar_data[0],
    #         "CL": polar_data_stall[1],
    #         "CD": polar_data_stall[2],
    #         "CL/CD": np.array(polar_data_stall[1]) / np.array(polar_data_stall[2]),
    #         "CS": polar_data_stall[3],
    #         "CMx": polar_data_stall[4],
    #         "CMy": polar_data_stall[5],
    #         "CMz": polar_data_stall[6],
    #     }
    # ).to_csv(path_to_csv, index=False)


def running_vsm_to_generate_csv_data(
    project_dir: str,
    vw: float,
    geom_scaling=6.5,
    height_correction_factor=1.0,  # 0.82,
    is_with_corrected_polar=True,
    mu=1.76e-5,
    reference_point=None,
    n_panels=150,
    spanwise_panel_distribution="uniform",
) -> None:
    if is_with_corrected_polar:
        print("Running VSM with corrected polar")
        name_appendix = "_corrected"
    else:
        print("Running VSM with breukels polar")
        name_appendix = "_breukels"

    vsm_input_path = Path(project_dir) / "data" / "vsm_input"
    # csv_file_path = Path(vsm_input_path) / "geometry_corrected.csv"
    # body_aero = create_body_aero(
    #     csv_file_path,
    #     n_panels,
    #     spanwise_panel_distribution,
    #     is_with_corrected_polar,
    #     vsm_input_path,
    #     geom_scaling,
    # )
    body_aero = BodyAerodynamics.from_file(
        Path(vsm_input_path) / "wing_geometry_from_CAD_orderded_tip_to_mid.csv",
        n_panels,
        spanwise_panel_distribution,
        is_with_corrected_polar=True,
        polar_data_dir=vsm_input_path,
        is_half_wing=True,
    )

    # ### Plotting reference point at mid-span plane
    # plt.figure()
    # n_half = len(rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_19ribs) // 2
    # LE = (
    #     rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_19ribs[n_half][0]
    #     / geom_scaling
    # )
    # TE = (
    #     rib_list_from_CAD_LE_TE_and_surfplan_d_tube_camber_19ribs[n_half][1]
    #     / geom_scaling
    # )
    # LE[2] = LE[2] * height_correction_factor
    # TE[2] = TE[2] * height_correction_factor
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

    # Solvers
    solver = Solver(reference_point=reference_point)
    # VSM_with_stall_correction = Solver(
    #     aerodynamic_model_type="VSM",
    #     is_with_artificial_damping=True,
    #     mu=mu,
    #     reference_point=reference_point,
    # )

    ### alpha sweep
    alphas_to_be_plotted = [
        -12.65,
        -7.15,
        -3.0,
        -2.25,
        2.35,
        4.75,
        6.75,
        8.8,
        10.95,
        11.95,
        12.8,
        14.0,
        15.75,
        17.85,
        19.75,
        22.55,
        24.0,
    ]
    alphas_to_be_plotted = np.linspace(-12.47, 23.56, 40)
    alphas_to_be_plotted = np.linspace(-5, 24, 30)

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
    betas_to_be_plotted = [
        # -20,
        # -14,
        # -12,
        # -10,
        # -8,
        # -6,
        # -4,
        # -2,
        0,
        2,
        4,
        6,
        8,
        12,
        14,
        20,
    ]
    betas_to_be_plotted = np.linspace(0, 10, 10)
    save_polar_data(
        angle_range=betas_to_be_plotted,
        angle_type="side_slip",
        angle_of_attack=6.8,
        name_appendix=name_appendix,
        body_aero=body_aero,
        solver=solver,
        solver_stall=None,
        vw=vw,
    )
    save_polar_data(
        angle_range=betas_to_be_plotted,
        angle_type="side_slip",
        angle_of_attack=11.9,
        name_appendix=name_appendix,
        body_aero=body_aero,
        solver=solver,
        solver_stall=None,
        vw=vw,
    )
    return


def main():

    ## scaled down geometry
    vw = 2.82
    geom_scaling = 6.5
    n_panels = 40

    # ## scaled down velocity
    # vw = 3.05
    # geom_scaling = 1.0

    ## Computing the reference point, to be equal as used for calc. the wind tunnel data Moments
    x_displacement_from_te = -0.157  # -0.172
    z_displacement_from_te = -0.252
    te_point_full_size = np.array([1.472144, 0, 3.696209])
    te_point_scaled = te_point_full_size / geom_scaling
    ## height was off even tho chord and span are matching perfectly...
    height_correction_factor = 1.0
    te_point_scaled[2] = te_point_scaled[2] * height_correction_factor
    reference_point = te_point_scaled + np.array(
        [x_displacement_from_te, 0, z_displacement_from_te]
    )
    print(f"reference_point: {reference_point}")
    # breakpoint()

    running_vsm_to_generate_csv_data(
        project_dir,
        vw=vw,
        is_with_corrected_polar=True,
        reference_point=reference_point,
        geom_scaling=geom_scaling,
        n_panels=n_panels,
    )
    # running_vsm_to_generate_csv_data(
    #     project_dir,
    #     vw=vw,
    #     is_with_corrected_polar=False,
    #     reference_point=reference_point,
    #     geom_scaling=geom_scaling,
    #     n_panels=n_panels,
    # )


if __name__ == "__main__":
    main()
