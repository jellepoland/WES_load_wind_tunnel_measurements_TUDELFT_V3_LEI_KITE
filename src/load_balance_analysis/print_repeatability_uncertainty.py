from pathlib import Path
import pandas as pd
import numpy as np
from pathlib import Path
from load_balance_analysis.functions_utils import project_dir, save_latex_table


def main(project_dir: Path) -> None:

    # Read the interpolation coefficients
    path_support_struc_aero_interp_coeffs = (
        Path(project_dir) / "data" / "support_struc_aero_interp_coeffs.csv"
    )
    support_struc_aero_interp_coeffs = pd.read_csv(
        path_support_struc_aero_interp_coeffs
    )

    # Read the labbook data
    path_labbook_double = Path(project_dir) / "data" / "labbook_repeatibility.csv"
    data = pd.read_csv(path_labbook_double, delimiter=";")
    data["measurement"] = data.groupby("Filename").cumcount()

    # Define the column names for the data files
    column_names = ["time", "Fx", "Fy", "Fz", "Mx", "My", "Mz", "error"]

    # Initialize an empty list to hold DataFrames
    dataframes = []

    # Read another data file and add information
    path_double_aoa_12_vw_20_unsteady = (
        Path(project_dir)
        / "data"
        / "repeatibility"
        / "double_aoa_12_vw_20_unsteady.txt"
    )
    df = pd.read_csv(
        path_double_aoa_12_vw_20_unsteady, delimiter="\t", names=column_names
    )  # Assuming tab-separated values for this example

    # Drop the 'error' column
    df = df.drop(columns=["error"])

    # Add a measurement column where each index is repeated every 19800 rows
    df["measurement"] = np.arange(len(df)) // 19800

    merged_df = pd.merge(data, df, on=["measurement"])
    merged_df.rename(columns={"vw": "vw_actual"}, inplace=True)
    merged_df["vw"] = np.ceil(merged_df["vw_actual"])
    merged_df = merged_df[merged_df["vw"] != 0]

    zerorun_ss_20 = np.array(
        [3.022004, 3.824555, 800.787406, 3.403333, 172.332697, 0.294079]
    )
    zerorun_ss_0 = np.array(
        [3.412569, 5.041744, 800.806084, 4.026951, 172.121360, 0.034760]
    )
    zerorun_ss_n20 = np.array(
        [3.337865, 6.318882, 800.781853, 4.693001, 172.152581, -0.232359]
    )

    cols = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
    new_cols = ["Cd", "Cs", "Cl", "Cmx", "Cmy", "Cmz"]

    # Create new columns for the zero run adjusted values
    for new_col, col in zip(new_cols, cols):
        merged_df[new_col] = merged_df[col]

    # Subtract the zero run values from the new columns
    for sideslip, zero_runs in zip(
        [-20, 0, 20], [zerorun_ss_n20, zerorun_ss_0, zerorun_ss_20]
    ):
        for new_col, col, zero_run in zip(new_cols, cols, zero_runs):
            merged_df.loc[merged_df["sideslip"] == sideslip, new_col] -= zero_run

    # Nondimensionalize the columns "Cd", "Cs", "Cl", "Cmx", "Cmy", and "Cmz"
    rho = merged_df["Density"]
    V = merged_df["vw_actual"]
    S_ref = 0.46
    c_ref = 0.4

    # Nondimensionalize the force columns
    merged_df["Cd"] /= 0.5 * rho * V**2 * S_ref
    merged_df["Cs"] /= 0.5 * rho * V**2 * S_ref
    merged_df["Cl"] /= 0.5 * rho * V**2 * S_ref

    # Nondimensionalize the moment columns
    merged_df["Cmx"] /= 0.5 * rho * V**2 * S_ref * c_ref
    merged_df["Cmy"] /= 0.5 * rho * V**2 * S_ref * c_ref
    merged_df["Cmz"] /= 0.5 * rho * V**2 * S_ref * c_ref

    # Select the support structure aerodynamic coefficients where the wind speed is the corresponding wind speed to this .lvm file
    aoa = merged_df["aoa"].unique()[0] - 7.25
    v = 20

    supp_coeffs = support_struc_aero_interp_coeffs[
        support_struc_aero_interp_coeffs["vw"] == 20
    ]
    for k in merged_df["sideslip"].unique():
        # select support structure aero coefficients for this sideslip
        c_s = supp_coeffs[supp_coeffs["sideslip"] == k]
        F_x = c_s.loc[c_s["channel"] == "Cx", ["a", "b", "c"]]
        F_y = c_s.loc[c_s["channel"] == "Cy", ["a", "b", "c"]]
        F_z = c_s.loc[c_s["channel"] == "Cz", ["a", "b", "c"]]
        M_x = c_s.loc[c_s["channel"] == "Cmx", ["a", "b", "c"]]
        M_y = c_s.loc[c_s["channel"] == "Cmy", ["a", "b", "c"]]
        M_z = c_s.loc[c_s["channel"] == "Cmz", ["a", "b", "c"]]

        # compute support structure aero coefficients for this wind speed, sideslip and angle of attack combination
        C_Fx_s = np.array(F_x["a"] * (aoa**2) + F_x["b"] * aoa + F_x["c"])[0]
        C_Fy_s = np.array(F_y["a"] * (aoa**2) + F_y["b"] * aoa + F_y["c"])[0]
        C_Fz_s = np.array(F_z["a"] * (aoa**2) + F_z["b"] * aoa + F_z["c"])[0]
        C_Mx_s = np.array(M_x["a"] * (aoa**2) + M_x["b"] * aoa + M_x["c"])[0]
        C_My_s = np.array(M_y["a"] * (aoa**2) + M_y["b"] * aoa + M_y["c"])[0]
        C_Mz_s = np.array(M_z["a"] * (aoa**2) + M_z["b"] * aoa + M_z["c"])[0]

        # subtract support structure aero coefficients for this wind speed, sideslip and aoa combination from merged_df
        merged_df.loc[
            (merged_df["sideslip"] == k) & (merged_df["vw"] == v), "Cd"
        ] -= C_Fx_s
        merged_df.loc[
            (merged_df["sideslip"] == k) & (merged_df["vw"] == v), "Cs"
        ] -= C_Fy_s
        merged_df.loc[
            (merged_df["sideslip"] == k) & (merged_df["vw"] == v), "Cl"
        ] -= C_Fz_s
        merged_df.loc[
            (merged_df["sideslip"] == k) & (merged_df["vw"] == v), "Cmx"
        ] -= C_Mx_s
        merged_df.loc[
            (merged_df["sideslip"] == k) & (merged_df["vw"] == v), "Cmy"
        ] -= C_My_s
        merged_df.loc[
            (merged_df["sideslip"] == k) & (merged_df["vw"] == v), "Cmz"
        ] -= C_Mz_s

    # Translate coordinate system
    # Parameters necessary to translate moments (aka determine position of cg)
    x_hinge = 441.5
    z_hinge = 1359
    l_cg = 625.4
    alpha_cg = np.deg2rad(merged_df["aoa"] - 23.82)
    x_cg = l_cg * np.cos(alpha_cg)
    z_cg = l_cg * np.sin(alpha_cg)
    A_side = 5.646
    A_ref = 19.753
    Cs_scale = A_ref / A_side

    merged_df["x_kite"] = (x_hinge + x_cg) / (1000 * c_ref)
    merged_df["z_kite"] = (z_hinge + z_cg) / (1000 * c_ref)

    # Rotation of coordinate system: force coefficients change
    merged_df["C_L"] = merged_df["Cl"] * -1
    merged_df["C_S"] = merged_df["Cs"] * -1
    merged_df["C_D"] = merged_df["Cd"]

    merged_df["C_roll"] = merged_df["Cmx"] - merged_df["Cs"] * merged_df["z_kite"]
    merged_df["C_pitch"] = (
        -merged_df["Cmy"]
        + merged_df["Cl"] * merged_df["x_kite"]
        - merged_df["Cd"] * merged_df["z_kite"]
    )
    merged_df["C_yaw"] = -merged_df["Cmz"] - merged_df["Cs"] * merged_df["x_kite"]

    ### correct for sideslip
    def correcting_for_sideslip(df: pd.DataFrame) -> pd.DataFrame:
        """
        [F_X_new]   [cos(β)  -sin(β)  0] [F_X_old]
        [F_Y_new] = [sin(β)   cos(β)  0] [F_Y_old]
        [F_Z_new]   [  0       0      1] [F_Z_old]
        """

        ## Grabbing sideslip array and converting to radians
        beta = np.deg2rad(df["sideslip"])
        # beta = np.deg2rad(10) * np.ones_like(df["sideslip"])

        ## Defining rotation matrix for each row
        def create_rotation_matrix(beta_angle):
            return np.array(
                [
                    [np.cos(beta_angle), np.sin(beta_angle), 0],
                    [-np.sin(beta_angle), np.cos(beta_angle), 0],
                    [0, 0, 1],
                ]
            )

        # Create arrays for forces and moments
        forces = np.array([df["C_D"], df["C_S"], df["C_L"]]).T
        moments = np.array([df["C_roll"], df["C_pitch"], df["C_yaw"]]).T

        # Initialize arrays for corrected forces and moments
        corrected_forces = np.zeros_like(forces)
        corrected_moments = np.zeros_like(moments)

        # Apply rotation to each row
        for i in range(len(df)):
            R = create_rotation_matrix(beta[i])
            corrected_forces[i] = R @ forces[i]
            corrected_moments[i] = R @ moments[i]

        # Update dataframe with corrected values
        df["C_D"], df["C_S"], df["C_L"] = corrected_forces.T
        df["C_roll"], df["C_pitch"], df["C_yaw"] = corrected_moments.T

        return df

    merged_df = correcting_for_sideslip(merged_df)

    # Define the coefficients and other relevant columns
    coefficients = ["Cd", "Cs", "Cl", "Cmx", "Cmy", "Cmz"]
    yaxis_names = ["C_D", "C_S", "C_L", "C_{roll}", "C_{pitch}", "C_{yaw}"]
    coefficients_plot_xaxis = ["C_D", "C_S", "C_L", "C_roll", "C_pitch", "C_yaw"]
    plot_titles = [
        "Drag coefficient",
        "Side-force coefficient",
        "Lift coefficient",
        "Roll moment coefficient",
        "Pitching moment coefficient",
        "Yawing moment coefficient",
    ]
    sideslip_angles = [-20, 0, 20]

    merged_df["measurement"] = (
        merged_df.groupby("sideslip").cumcount() % 3 + 1
    )  # Measurement labels: 1, 2, 3

    # for sideslip in sideslip_angles:
    #     print(f"\nSideslip angle: {sideslip}°")
    #     df_local = merged_df[merged_df["sideslip"] == sideslip]
    #     df_local_m1 = df_local[df_local["measurement"] == 1]
    #     df_local_m2 = df_local[df_local["measurement"] == 2]
    #     df_local_m3 = df_local[df_local["measurement"] == 3]
    #     for coeff in coefficients_plot_xaxis:
    #         # print(
    #         #     f"coeff: {coeff}, m1_mean: {df_local_m1[coeff].mean()}, m2_mean: {df_local_m2[coeff].mean()}, m3_mean: {df_local_m3[coeff].mean()}"
    #         # )
    #         print(
    #             f"coeff: {coeff}, std of mean: {1e4*np.std([df_local_m1[coeff].mean(), df_local_m2[coeff].mean(), df_local_m3[coeff].mean()]):.3f} x 1e-4"
    #         )

    # Coefficients to be used
    coefficients_plot_xaxis = ["C_L", "C_D", "C_S", "C_roll", "C_pitch", "C_yaw"]

    # Assuming merged_df is your DataFrame containing the data with the appropriate columns
    # Create a DataFrame to hold the formatted data
    formatted_data = []

    # Loop over sideslip angles and coefficients
    for sideslip in sideslip_angles:
        # Initialize a row for each sideslip angle
        row = {r"Sideslip angle ($\beta$)": rf"{sideslip} \unit{{\degree}}"}

        # Filter data for the current sideslip angle
        df_local = merged_df[merged_df["sideslip"] == sideslip]
        df_local_m1 = df_local[df_local["measurement"] == 1]
        df_local_m2 = df_local[df_local["measurement"] == 2]
        df_local_m3 = df_local[df_local["measurement"] == 3]

        # For each coefficient, calculate the standard deviation of the mean
        for coeff in coefficients_plot_xaxis:
            std_of_mean = 1e4 * np.std(
                [
                    df_local_m1[coeff].mean(),
                    df_local_m2[coeff].mean(),
                    df_local_m3[coeff].mean(),
                ]
            )
            row[coeff] = f"{std_of_mean:.3f}"

        # Append the row to the formatted data list
        formatted_data.append(row)

    # Convert the formatted data into a DataFrame
    df_table = pd.DataFrame(formatted_data)

    # Save the table
    save_latex_table(
        df_table,
        Path(project_dir)
        / "results"
        / "tables"
        / "repeatibility_standard_deviation.tex",
    )

    # Show the table
    print(f"\n--- Repeatability uncertainty table ---\n")
    print(df_table.to_string(index=False))


if __name__ == "__main__":
    main(project_dir)
