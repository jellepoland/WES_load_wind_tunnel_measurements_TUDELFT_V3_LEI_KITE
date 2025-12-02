import numpy as np
import pandas as pd
from pathlib import Path


def read_csv_with_locale(file_path: Path) -> pd.DataFrame:
    """
    Reads a CSV file handling different locale decimal separators.

    Parameters:
    file_path (str): Path to the CSV file
    dtypes (dict): Dictionary of column data types

    Returns:
    pandas.DataFrame: The loaded and corrected dataframe
    """
    # Specifying the datatypes of each column
    dtypes = {
        "Filename": "str",
        "vw": "float64",
        "aoa": "float64",
        "sideslip": "float64",
        "Date": "str",
        "F_X": "float64",
        "F_Y": "float64",
        "F_Z": "float64",
        "M_X": "float64",
        "M_Y": "float64",
        "M_Z": "float64",
        "Dpa": "float64",
        "Pressure": "float64",
        "Temp": "float64",
        "Density": "float64",
    }

    # First, try to read the CSV with a decimal handler
    try:
        df = pd.read_csv(
            file_path,
            dtype={
                col: "str" for col in dtypes.keys()
            },  # Read everything as string first
            thousands=None,  # Disable thousands separator handling
        )

        # Function to safely convert string to float
        def safe_float_convert(x):
            if pd.isna(x):
                return np.nan
            try:
                return float(x)
            except ValueError:
                # Try replacing comma with period
                try:
                    return float(x.replace(",", "."))
                except:
                    return np.nan

        # Convert numeric columns
        for col, dtype in dtypes.items():
            if dtype == "float64" and col in df.columns:
                df[col] = df[col].apply(safe_float_convert)

        return df

    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        raise


def nondimensionalize(df: pd.DataFrame, S_ref: float, c_ref: float) -> pd.DataFrame:
    """
    Nondimensionalize the force and moment columns in the dataframe

    Input:
        df (pandas.DataFrame): The input dataframe
        S_ref (float): Reference area
        c_ref (float): Reference chord length

    Output:
        pandas.DataFrame: The dataframe with the force and moment columns nondimensionalized
    """
    rho = df["Density"]
    V = df["vw"]

    # Nondimensionalize the force columns
    df["CF_X"] = df["F_X"] / (0.5 * rho * V**2 * S_ref)
    df["CF_Y"] = df["F_Y"] / (0.5 * rho * V**2 * S_ref)
    df["CF_Z"] = df["F_Z"] / (0.5 * rho * V**2 * S_ref)

    # Nondimensionalize the moment columns
    df["CM_X"] = df["M_X"] / (0.5 * rho * V**2 * S_ref * c_ref)
    df["CM_Y"] = df["M_Y"] / (0.5 * rho * V**2 * S_ref * c_ref)
    df["CM_Z"] = df["M_Z"] / (0.5 * rho * V**2 * S_ref * c_ref)

    ## Also do this for the raw columns, to later calculate SNR
    df["CF_X_raw"] = df["F_X_raw"] / (0.5 * rho * V**2 * S_ref)
    df["CF_Y_raw"] = df["F_Y_raw"] / (0.5 * rho * V**2 * S_ref)
    df["CF_Z_raw"] = df["F_Z_raw"] / (0.5 * rho * V**2 * S_ref)
    df["CM_X_raw"] = df["M_X_raw"] / (0.5 * rho * V**2 * S_ref * c_ref)
    df["CM_Y_raw"] = df["M_Y_raw"] / (0.5 * rho * V**2 * S_ref * c_ref)
    df["CM_Z_raw"] = df["M_Z_raw"] / (0.5 * rho * V**2 * S_ref * c_ref)

    return df


# function that translates coordinate system
def translate_from_origin_to_cg(
    df: pd.DataFrame,
    x_hinge: float,
    z_hinge: float,
    l_cg: float,
    alpha_cg_delta_with_rod: float,
) -> pd.DataFrame:

    ##TODO: pre 23-10-2025 ##

    ##TODO: update alpha_cg_comes_in_deg should be rad
    # alpha_cg = np.deg2rad(df["aoa"] - alpha_cg_delta_with_rod)
    # x_cg = l_cg * np.cos(alpha_cg)
    # z_cg = l_cg * np.sin(alpha_cg)
    # x_hcg = (x_hinge + x_cg) / (1000 * c_ref)
    # z_hcg = (z_hinge + z_cg) / (1000 * c_ref)

    ### Calculations for origin movement to cg
    alpha_cg = np.deg2rad(df["aoa"]) - np.deg2rad(alpha_cg_delta_with_rod)
    x_hinge_to_cg_mm = l_cg * np.cos(alpha_cg)
    z_hinge_to_cg_mm = l_cg * np.sin(alpha_cg)
    x_cg_to_origin_mm = x_hinge + x_hinge_to_cg_mm
    z_cg_to_origin_mm = z_hinge + z_hinge_to_cg_mm
    # converting to meters
    x_cg_to_origin = x_cg_to_origin_mm / 1000
    z_cg_to_origin = z_cg_to_origin_mm / 1000

    # for x, z in zip(x_cg_to_origin, z_cg_to_origin):
    #     print(f"x_delta: {x:.2f}, z_delta: {z:.2f}")

    # print(f"\nCalculating the translation of the coordinate system")
    # print(f" x_hinge = {x_cg_to_origin} m")
    # print(f" z_hinge = {z_cg_to_origin} m")

    # enforcing the right signs
    # this must be represented in the OLD coordinate system
    # "That is literally the vector from the old origin to the CG in the old coordinate system.""
    # x_cg_to_origin = -np.abs(x_cg_to_origin)
    # z_cg_to_origin = -np.abs(z_cg_to_origin)

    # print(f"after sign correction")
    # print(f" x_hinge = {x_cg_to_origin} m")
    # print(f" z_hinge = {z_cg_to_origin} m")

    # # rotation of coordinate system: force coefficients change
    # df["C_D"] = df["CF_X"]
    # df["C_S"] = df["CF_Y"] * -1
    # df["C_L"] = df["CF_Z"] * -1

    # # rotation of coordinate system: moment coefficients change
    # # df["C_roll"] = df["CM_X"] - df["CF_Y"] * z_cg_to_origin
    # # df["C_pitch"] = (
    # #     -df["CM_Y"] + df["CF_Z"] * x_cg_to_origin - df["CF_X"] * z_cg_to_origin
    # # )
    # # df["C_yaw"] = -df["CM_Z"] - df["CF_Y"] * x_cg_to_origin

    # df["C_roll"] = df["CM_X"] - z_cg_to_origin * df["CF_Y"]
    # df["C_pitch"] = (
    #     -df["CM_Y"] - z_cg_to_origin * df["CF_X"] + x_cg_to_origin * df["CF_Z"]
    # )
    # df["C_yaw"] = -df["CM_Z"] - x_cg_to_origin * df["CF_Y"]

    # # Suppose x_cg_to_origin, z_cg_to_origin are scalar floats
    # # describing the translation from the old origin (bottom-right)
    # # to the new origin (top-left, c.g.) in the OLD coordinate frame.
    # x_cg_to_origin = 1.23
    # z_cg_to_origin = 4.56

    # 1) Define the shift vector r (from old origin O to cg in OLD reference frame')
    # r = np.array(x_cg_to_origin, 0.0, z_cg_to_origin)
    r = np.column_stack([x_cg_to_origin, np.zeros_like(x_cg_to_origin), z_cg_to_origin])

    # print(f"r: {r}")
    # 3) Pull out the old forces & moments as NumPy arrays of shape (N,3)
    F_origin = df[["F_X", "F_Y", "F_Z"]].values  # shape (N, 3)
    M_origin = df[["M_X", "M_Y", "M_Z"]].values  # shape (N, 3)

    # M_due_to_forces = np.cross(r, F_old, axis=0)

    # M_due_to_forces_list = []
    # for r, F, M in zip(r, F_origin, M_origin):
    #     print(f"\nr: {r}")
    #     print(f"F: {F}")
    #     print(f"M: {M}")
    #     M_due_to_forces = np.cross(r, F)
    #     print(f"M_new: {M_due_to_forces}")
    #     M_due_to_forces_list.append(M_due_to_forces)

    # 4) Compute the moment about the c.g. in the OLD axes:
    #    M_cg_old = M_old + r x F_old
    #    np.cross(r, F_old) will broadcast r (shape (3,)) across each row of F_old
    M_cg = M_origin + np.cross(r, F_origin)

    # 5) Transform forces & moments into the NEW axes via matrix multiplication.
    #    Because each row is a vector, we multiply on the right by T^T or T
    #    (for a diagonal matrix T, T^T == T anyway).
    #   Define the transformation matrix T that flips y and z:
    #    old vector [x, y, z] becomes [ x, -y, -z ]
    # T = np.diag([1.0, -1.0, -1.0])  # 3x3
    # F_new = F_old @ T
    # M_new = M_cg_old @ T

    # # 6) Write results back into the dataframe.
    # #    For example, rename them as (C_D, C_S, C_L) and (C_roll, C_pitch, C_yaw).
    # df["C_D"] = F_new[:, 0]
    # df["C_S"] = F_new[:, 1]
    # df["C_L"] = F_new[:, 2]

    # df["C_roll"] = M_new[:, 0]
    # df["C_pitch"] = M_new[:, 1]
    # df["C_yaw"] = M_new[:, 2]

    df["F_X"] = df["F_X"]
    df["F_Y"] = df["F_Y"]
    df["F_Z"] = df["F_Z"]
    df["M_X"] = M_cg[:, 0]
    df["M_Y"] = M_cg[:, 1]
    df["M_Z"] = M_cg[:, 2]

    # ##TODO: post 23-10-2025
    # # Convert hinge location from millimetres (lab reference) to metres (model reference)
    # x_hinge_m = x_hinge / 1000.0
    # z_hinge_m = z_hinge / 1000.0

    # # Geometry parameters for the TU Delft tow-point rig
    # alpha_rod_to_chordline = 6.3  # deg
    # l_rod = 0.415  # m
    # z_TE_fullscale = 11.004972573823276  # m
    # x_TE_fullscale = 1.443146003226444  # m
    # geom_scaling_factor = 6.5

    # alpha_deg = df["aoa"] - alpha_rod_to_chordline
    # x_alpha, z_alpha, _, _ = compute_tow_point(
    #     alpha_deg,
    #     alpha_rod_to_chordline=alpha_rod_to_chordline,
    #     l_rod=l_rod,
    #     x_hinge=x_hinge_m,
    #     z_hinge=z_hinge_m,
    #     z_TE_fullscale=z_TE_fullscale,
    #     x_TE_fullscale=x_TE_fullscale,
    #     geom_scaling_factor=geom_scaling_factor,
    # )
    # # the z-coordinate must be inverted because wind tunnel has z pointing downwards
    # r = np.column_stack([x_alpha, np.zeros_like(x_alpha), -z_alpha])
    # # defining the forces
    # F_origin = df[["F_X", "F_Y", "F_Z"]].values  # shape (N, 3)
    # M_origin = df[["M_X", "M_Y", "M_Z"]].values  # shape (N, 3)
    # # computing moment
    # M_tow_point = M_origin + np.cross(r, F_origin)
    # # writing back to dataframe
    # df["F_X"] = df["F_X"]
    # df["F_Y"] = df["F_Y"]
    # df["F_Z"] = df["F_Z"]
    # df["M_X"] = M_tow_point[:, 0]
    # df["M_Y"] = M_tow_point[:, 1]
    # df["M_Z"] = M_tow_point[:, 2]

    return df


def rotate_axis_at_origin(df: pd.DataFrame) -> pd.DataFrame:

    F_old = df[["F_X", "F_Y", "F_Z"]].values  # shape (N, 3)
    M_old = df[["M_X", "M_Y", "M_Z"]].values  # shape (N, 3)
    T = np.diag([1.0, -1.0, -1.0])  # 3x3
    F_new = F_old @ T
    M_new = M_old @ T

    # 6) Write results back into the dataframe.
    #    For example, rename them as (C_D, C_S, C_L) and (C_roll, C_pitch, C_yaw).
    df["F_X"] = F_new[:, 0]
    df["F_Y"] = F_new[:, 1]
    df["F_Z"] = F_new[:, 2]
    df["M_X"] = M_new[:, 0]
    df["M_Y"] = M_new[:, 1]
    df["M_Z"] = M_new[:, 2]

    return df


def correcting_for_sideslip(df: pd.DataFrame) -> pd.DataFrame:
    """
    Suppose the turntable measured beta positive clockwise,
    but we want it positive counterclockwise in our final x,y,z frame.
    """
    # Convert to radians
    ##TODO: added the - on 02/12/2025, I think this is correct now. It was needed to get the C_D in check.
    beta = -np.deg2rad(df["sideslip"])

    # Create the rotation matrix
    def create_rotation_matrix(beta_angle):
        return np.array(
            [
                [np.cos(beta_angle), -np.sin(beta_angle), 0],
                [np.sin(beta_angle), np.cos(beta_angle), 0],
                [0, 0, 1],
            ]
        )

    # Forces and moments as arrays
    forces = np.array([df["F_X"], df["F_Y"], df["F_Z"]]).T
    moments = np.array([df["M_X"], df["M_Y"], df["M_Z"]]).T

    # Apply rotation row by row
    corrected_forces = np.zeros_like(forces)
    corrected_moments = np.zeros_like(moments)

    for i in range(len(df)):
        R = create_rotation_matrix(beta[i])
        corrected_forces[i] = R @ forces[i]
        corrected_moments[i] = R @ moments[i]

    # Put corrected values back
    df["F_X"], df["F_Y"], df["F_Z"] = corrected_forces.T
    df["M_X"], df["M_Y"], df["M_Z"] = corrected_moments.T

    return df


# def substract_support_structure_aero_coefficients_1_line(
#     df: pd.DataFrame, support_struc_aero_interp_coeffs_path: Path
# ) -> pd.DataFrame:

#     # reading the interpolated coefficients
#     support_struc_aero_interp_coeffs = pd.read_csv(
#         support_struc_aero_interp_coeffs_path
#     )
#     # print(f"columns: {merged_df.columns}")
#     # Select the support structure aerodynamic coefficients where the wind speed is the corresponding wind speed to this .lvm file
#     cur_vw = df["vw"].unique()[0]
#     # print(f"cur_vw: {np.around(cur_vw)}")
#     # print(f'support_struc_aero_interp_coeffs["vw"]: {support_struc_aero_interp_coeffs["vw"].unique()}')
#     supp_coeffs = support_struc_aero_interp_coeffs[
#         support_struc_aero_interp_coeffs["vw"] == int(np.around(cur_vw))
#     ]
#     # print(f"supp_coeffs: {supp_coeffs}")
#     aoa_kite = df["aoa_kite"].unique()[0]

#     for k in df["sideslip"].unique():
#         # print(f"k: {k}")
#         # select support structure aero coefficients for this sideslip
#         c_s = supp_coeffs[supp_coeffs["sideslip"] == k]
#         F_x = c_s.loc[c_s["channel"] == "Cx", ["a", "b", "c"]]
#         F_y = c_s.loc[c_s["channel"] == "Cy", ["a", "b", "c"]]
#         F_z = c_s.loc[c_s["channel"] == "Cz", ["a", "b", "c"]]
#         M_x = c_s.loc[c_s["channel"] == "Cmx", ["a", "b", "c"]]
#         M_y = c_s.loc[c_s["channel"] == "Cmy", ["a", "b", "c"]]
#         M_z = c_s.loc[c_s["channel"] == "Cmz", ["a", "b", "c"]]

#         # compute support structure aero coefficients for this wind speed, sideslip and angle of attack combination
#         C_Fx_s = np.array(F_x["a"] * (aoa_kite**2) + F_x["b"] * aoa_kite + F_x["c"])[0]
#         C_Fy_s = np.array(F_y["a"] * (aoa_kite**2) + F_y["b"] * aoa_kite + F_y["c"])[0]
#         C_Fz_s = np.array(F_z["a"] * (aoa_kite**2) + F_z["b"] * aoa_kite + F_z["c"])[0]
#         C_Mx_s = np.array(M_x["a"] * (aoa_kite**2) + M_x["b"] * aoa_kite + M_x["c"])[0]
#         C_My_s = np.array(M_y["a"] * (aoa_kite**2) + M_y["b"] * aoa_kite + M_y["c"])[0]
#         C_Mz_s = np.array(M_z["a"] * (aoa_kite**2) + M_z["b"] * aoa_kite + M_z["c"])[0]

#         # subtract support structure aero coefficients for this wind speed, sideslip and aoa combination from merged_df
#         df.loc[df["sideslip"] == k, "C_D"] -= C_Fx_s
#         df.loc[df["sideslip"] == k, "C_S"] -= C_Fy_s
#         df.loc[df["sideslip"] == k, "C_L"] -= C_Fz_s
#         df.loc[df["sideslip"] == k, "C_roll"] -= C_Mx_s
#         df.loc[df["sideslip"] == k, "C_pitch"] -= C_My_s
#         df.loc[df["sideslip"] == k, "C_yaw"] -= C_Mz_s

#     return df


def substract_support_structure_aero_coefficients(
    df: pd.DataFrame, support_struc_aero_interp_coeffs_path: Path
) -> pd.DataFrame:

    # Read the interpolated coefficients from the CSV file
    support_struc_aero_interp_coeffs = pd.read_csv(
        support_struc_aero_interp_coeffs_path
    )

    # print(f"columns: {support_struc_aero_interp_coeffs.columns}")

    # Select the support structure aerodynamic coefficients where the wind speed is the corresponding wind speed to this .lvm file
    cur_vw = df["vw"].unique()[0]
    supp_coeffs = support_struc_aero_interp_coeffs[
        support_struc_aero_interp_coeffs["vw"] == int(np.around(cur_vw))
    ]
    # Retrieve the angle of attack for the kite
    aoa_kite = df["aoa_kite"].unique()[0]

    if supp_coeffs.empty:
        raise ValueError(
            f"No support structure aerodynamic coefficients available for vw={cur_vw}."
        )

    tol_sideslip = 0.51  # tolerate small offsets introduced during zigzag corrections

    for k in df["sideslip"].unique():
        # Select support structure aero coefficients for this sideslip
        mask = np.isclose(supp_coeffs["sideslip"], k, atol=tol_sideslip)
        c_s = supp_coeffs[mask]

        if c_s.empty:
            nearest_idx = (supp_coeffs["sideslip"] - k).abs().idxmin()
            nearest_sideslip = supp_coeffs.loc[nearest_idx, "sideslip"]
            print(
                f"Warning: sideslip {k:.2f} missing in support coefficients; "
                f"using nearest sideslip {nearest_sideslip:.2f}."
            )
            c_s = supp_coeffs[supp_coeffs["sideslip"] == nearest_sideslip]

        required_channels = {"C_D", "C_S", "C_L", "C_roll", "C_pitch", "C_yaw"}
        missing_channels = required_channels.difference(set(c_s["channel"]))
        if missing_channels:
            raise ValueError(
                f"Support structure coefficients missing channels {sorted(missing_channels)} "
                f"for vw={cur_vw}, sideslip≈{k:.2f}."
            )

        def evaluate_channel(channel_name: str) -> float:
            row = c_s[c_s["channel"] == channel_name]
            if row.empty:
                raise ValueError(
                    f"Support structure coefficients missing channel {channel_name} "
                    f"for vw={cur_vw}, sideslip≈{k:.2f}."
                )
            row = row.iloc[0]
            degree = int(row.get("degree", 0))
            coeffs = [float(row.get(f"coef_{i}", 0.0)) for i in range(degree + 1)]
            return np.polyval(coeffs[::-1], aoa_kite)

        C_Fx_s = evaluate_channel("C_D")
        C_Fy_s = evaluate_channel("C_S")
        C_Fz_s = evaluate_channel("C_L")
        C_Mx_s = evaluate_channel("C_roll")
        C_My_s = evaluate_channel("C_pitch")
        C_Mz_s = evaluate_channel("C_yaw")

        # Subtract support structure aero coefficients from the main dataframe for this wind speed, sideslip, and aoa combination
        df.loc[df["sideslip"] == k, "C_D"] -= C_Fx_s
        df.loc[df["sideslip"] == k, "C_S"] -= C_Fy_s
        df.loc[df["sideslip"] == k, "C_L"] -= C_Fz_s
        df.loc[df["sideslip"] == k, "C_roll"] -= C_Mx_s
        df.loc[df["sideslip"] == k, "C_pitch"] -= C_My_s
        df.loc[df["sideslip"] == k, "C_yaw"] -= C_Mz_s

        # add the support structure aero coefficients to the dataframe
        df.loc[df["sideslip"] == k, "C_D_s"] = C_Fx_s
        df.loc[df["sideslip"] == k, "C_S_s"] = C_Fy_s
        df.loc[df["sideslip"] == k, "C_L_s"] = C_Fz_s
        df.loc[df["sideslip"] == k, "C_roll_s"] = C_Mx_s
        df.loc[df["sideslip"] == k, "C_pitch_s"] = C_My_s
        df.loc[df["sideslip"] == k, "C_yaw_s"] = C_Mz_s

    return df


def add_dyn_visc_and_reynolds(
    df: pd.DataFrame,
    delta_celsius_kelvin: float,
    mu_0: float,
    T_0: float,
    C_suth: float,
    c_ref: float,
) -> pd.DataFrame:
    # add dynamic viscosity and reynolds number column
    T = df["Temp"] + delta_celsius_kelvin

    dynamic_viscosity = (
        mu_0 * (T / T_0) ** 0.5 * (T_0 + C_suth) / (T + C_suth)
    )  # sutherland's law
    df["Rey"] = df["Density"] * df["vw"] * c_ref / dynamic_viscosity

    return df


def computing_kite_cg(x_hinge, z_hinge, l_cg, alpha_cg_delta_with_rod):

    ### OLD measurements ###
    S_ref = 0.46
    c_ref = 0.4

    # parameters necessary to translate moments (aka determine position of cg)
    x_hinge = (
        441.5  # x distance between force balance coord. sys. and hinge point in mm
    )
    z_hinge = 1359  # z distance between force balance coord. sys. and hinge point in mm
    l_cg = 625.4  # distance between hinge point and kite CG
    alpha_cg_delta_with_rod = 23.82
    delta_celsius_kelvin = 273.15
    mu_0 = 1.716e-5
    T_0 = 273.15
    C_suth = 110.4
    delta_aoa_rod_to_alpha = 6.3  # 7.25
    l_rod = 400  # length of the rod from hinge to TE

    ### NEW measurements ###
    x_hinge = -(395 + 39)
    z_hinge = -(1185 + 168.5)  # 168.5 is the balance internal height
    l_rod = 415

    print(f"\nThe calculation of the center of gravity")
    print(
        f" hinge point is located at: \nx_hinge = {x_hinge} mm\nz_hinge = {z_hinge} mm"
    )
    print(f" distance between hinge point and kite CG: l_cg = {l_cg} mm")
    print(
        f" angle between the rod and the kite: alpha_cg_delta_with_rod = {alpha_cg_delta_with_rod} deg"
    )
    angle_cg = 10 - alpha_cg_delta_with_rod
    print(
        f"\n For an measured aoa of 10deg, this would make angle_cg = 10 - alpha_cg_delta_with_rod = {angle_cg} deg"
    )
    x_hinge_cg = l_cg * np.cos(np.deg2rad(angle_cg))
    print(f"x_hinge,cg = l_cg * np.cos(angle_cg) = {x_hinge_cg:.2f} mm")
    z_hinge_cg = l_cg * np.sin(np.deg2rad(angle_cg))
    print(f"z_hinge,cg = l_cg * np.sin(angle_cg) = {z_hinge_cg:.2f} mm")
    x_cg_origin = x_hinge - x_hinge_cg
    print(f"\n x_cg,origin = x_hinge + x_hinge,cg = {x_cg_origin:.2f} mm")
    z_cg_origin = z_hinge + z_hinge_cg
    print(f" z_cg,origin = z_hinge + z_hinge,cg = {z_cg_origin:.2f} mm")
    print(f"---------------------------------")
    print(f"\n Location of TE, with a 0 measured aoa")
    x_te = x_hinge - l_rod
    print(
        f" x_te = x_hinge + l_rod = {x_te:.2f} mm, with l_rod (distance from hinge-to-TE) = {l_rod:.2f} mm"
    )
    z_te = z_hinge
    print(f" z_te = z_hinge = {z_te:.2f} mm")
    angle_cg = 0 - alpha_cg_delta_with_rod
    print(
        f" angle cg with 0 deg aoa: angle_cg = 0 - alpha_cg_delta_with_rod = {angle_cg:.2f} deg"
    )
    x_cg_origin = x_hinge - l_cg * np.cos(np.deg2rad(angle_cg))
    print(f"\nx_cg,origin = {x_cg_origin:.2f} mm")
    print(f"z_cg,origin = {z_cg_origin:.2f} mm")
    z_cg_origin = z_hinge + l_cg * np.sin(np.deg2rad(angle_cg))
    print(f"\n Distance between cg and TE at 0 deg aoa")
    x_te_cg = x_cg_origin - x_te
    print(f" x_te_cg = x_cg,origin - x_te = {x_te_cg:.2f} mm")
    z_te_cg = z_cg_origin - z_te
    print(f" z_te_cg = z_cg,origin - z_te = {z_te_cg:.2f} mm")
    print(f"alternative x_te_cg: {l_cg * np.cos(np.deg2rad(angle_cg)) - l_rod:.2f}")
    print(f"alternative z_te_cg: {l_cg * np.sin(np.deg2rad(angle_cg)):.2f}")

    ## Checking the calculation method
    alpha_cg = np.deg2rad(0) - np.deg2rad(alpha_cg_delta_with_rod)
    x_hinge_to_cg_mm = l_cg * np.cos(alpha_cg)
    z_hinge_to_cg_mm = l_cg * np.sin(alpha_cg)
    x_cg_to_origin_mm = x_hinge - x_hinge_to_cg_mm
    z_cg_to_origin_mm = z_hinge + z_hinge_to_cg_mm
    # converting to meters
    x_cg_to_origin = x_cg_to_origin_mm / 1000
    z_cg_to_origin = z_cg_to_origin_mm / 1000

    print(f"\nChecking the calculation method")
    print(f"x_hinge_to_cg_mm: {x_hinge_to_cg_mm:.2f} mm")
    print(f"z_hinge_to_cg_mm: {z_hinge_to_cg_mm:.2f} mm")
    print(f"x_cg_to_origin_mm: {x_cg_to_origin_mm:.2f} mm")
    print(f"z_cg_to_origin_mm: {z_cg_to_origin_mm:.2f} mm")
    print(f"x_cg_to_origin: {x_cg_to_origin:.2f} m")
    print(f"z_cg_to_origin: {z_cg_to_origin:.2f} m")

    ## plotting a 2D plot with all the points
    from matplotlib import pyplot as plt

    ## flipping all the x values
    # x_cg_origin = -x_cg_origin
    # x_te = -x_te

    plt.figure()
    plt.plot(x_hinge, z_hinge, "ro", label="hinge point")
    plt.plot(x_cg_origin, z_cg_origin, "yo", label="CG")
    plt.plot(x_te, z_te, "bo", label="TE")
    plt.plot((x_te - 395), z_te, "bx", label="LE (approximate)")
    plt.plot(0, 0, "go", label="origin")
    plt.legend()
    plt.axis("equal")
    plt.grid()
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(x_cg_origin, z_cg_origin, "yo", label="CG")
    plt.plot(x_te, z_te, "bo", label="TE")
    plt.plot((x_te - 395), z_te, "bx", label="LE (approximate)")
    plt.legend()
    plt.axis("equal")
    plt.grid()
    plt.show()
    plt.close()

    breakpoint()


def compute_tow_point(
    alpha_deg,
    alpha_rod_to_chordline=6.3,  # deg
    l_rod=0.415,
    x_hinge: float = 0.434,
    z_hinge: float = 1.3535,
    z_TE_fullscale=11.004972573823276,
    x_TE_fullscale=1.443146003226444,
    geom_scaling_factor=6.5,
):

    delta_x_tow_point = x_TE_fullscale / geom_scaling_factor + l_rod * np.cos(
        np.deg2rad(alpha_rod_to_chordline)
    )
    delta_z_tow_point = z_TE_fullscale / geom_scaling_factor
    phi_0 = np.arctan(delta_z_tow_point / delta_x_tow_point)
    r_tow_to_hinge = np.sqrt(delta_x_tow_point**2 + delta_z_tow_point**2)

    x_alpha = -r_tow_to_hinge * np.cos(phi_0 - np.deg2rad(alpha_deg)) - x_hinge
    z_alpha = -r_tow_to_hinge * np.sin(phi_0 - np.deg2rad(alpha_deg)) + z_hinge

    return x_alpha, z_alpha, delta_x_tow_point, delta_z_tow_point


def apply_wind_tunnel_corrections(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply wind-tunnel corrections to the aerodynamic coefficients and angles.

    ANGLES are corrected in:
    - plot_literature_saving.py
    - plot_reynolds_variation.py
    """
    df_corr = df.copy()

    # # Correct angle of attack (aoa_kite)
    # if "aoa_kite" in df_corr.columns and "C_L" in df_corr.columns:
    #     df_corr["aoa_kite"] = df_corr["aoa_kite"] + (-0.818 * df_corr["C_L"])

    # # Correct sideslip
    # if "sideslip" in df_corr.columns and "C_S" in df_corr.columns:
    #     df_corr["sideslip"] = df_corr["sideslip"] + (-0.72 * df_corr["C_S"])

    # Correct drag coefficient (C_D)
    if (
        "C_D" in df_corr.columns
        and "C_L" in df_corr.columns
        and "C_S" in df_corr.columns
    ):
        df_corr["C_D"] = df_corr["C_D"] + (
            -0.0078 * df_corr["C_L"] ** 2 - 0.0078 * df_corr["C_S"] ** 2
        )

    # # Correct lift coefficient (C_L)
    # if "C_L" in df_corr.columns:
    #     df_corr["C_L"] = df_corr["C_L"] + (0.025 * df_corr["C_L"])

    # # Correct side-force coefficient (C_S)
    # if "C_S" in df_corr.columns:
    #     df_corr["C_S"] = df_corr["C_S"] + (0.003 * df_corr["C_S"])

    # # Correct pitching moment coefficient (C_pitch)
    # if "C_pitch" in df_corr.columns and "C_L" in df_corr.columns:
    #     df_corr["C_pitch"] = df_corr["C_pitch"] + (-0.0053 * df_corr["C_L"])

    # # Correct yawing moment coefficient (C_yaw)
    # if "C_yaw" in df_corr.columns and "C_S" in df_corr.columns:
    #     df_corr["C_yaw"] = df_corr["C_yaw"] + (-0.0008 * df_corr["C_S"])

    return df_corr


def processing_raw_lvm_data_into_csv(
    folder_dir: Path,
    is_kite: bool,
    is_zigzag: bool,
    support_struc_aero_interp_coeffs_path: Path,
    S_ref: float = 0.46,
    c_ref: float = 0.395,
    x_hinge: float = 434,
    z_hinge: float = 1353.5,
    l_cg: float = 625.4,
    alpha_cg_delta_with_rod: float = 23.82,
    delta_celsius_kelvin: float = 273.15,
    mu_0: float = 1.716e-5,
    T_0: float = 273.15,
    C_suth: float = 110.4,
    delta_aoa_rod_to_alpha: float = 6.3,
):
    # TODO: toggle this on/off
    ### computing_kite_cg(x_hinge, z_hinge, l_cg, alpha_cg_delta_with_rod)
    # alpha_rod_to_chordline = 6.3  # deg
    # l_rod = 0.415
    # x_hinge = 0.434
    # z_hinge = 1.3535
    # z_TE_fullscale = 11.004972573823276
    # x_TE_fullscale = 1.443146003226444
    # geom_scaling_factor = 6.5
    # x_alpha_0, z_alpha_0, _, _ = compute_tow_point(alpha=0)
    # x_alpha_10, z_alpha_10, delta_x_tow_point, delta_z_tow_point = compute_tow_point(
    #     alpha=10
    # )

    # from matplotlib import pyplot as plt

    # plt.figure()
    # plt.plot(0, 0, "ro", label="origin")
    # plt.plot(-x_hinge, z_hinge, "go", label="hinge point")
    # plt.plot(
    #     -delta_x_tow_point - x_hinge,
    #     -delta_z_tow_point + z_hinge,
    #     "x",
    #     label="tow point initial",
    # )
    # plt.plot(
    #     -(x_TE_fullscale / geom_scaling_factor) - x_hinge,
    #     0 + z_hinge,
    #     "x",
    #     label="TE full-scale",
    # )
    # plt.plot(x_alpha_0, z_alpha_0, "bo", label="tow point at 0 deg aoa")
    # plt.plot(x_alpha_10, z_alpha_10, "co", label="tow point at 10 deg aoa")
    # plt.legend()
    # plt.axis("equal")
    # plt.grid()
    # plt.show()
    # breakpoint()

    if is_kite:
        print(f"PROCESSING KITE, substracting support-structure")
    else:
        print(f"PROCESSING SUPPORT-STRUCTURE")

    if is_zigzag:
        print(f"Processing ZIGZAG")

    files = sorted(
        folder_dir.iterdir(), key=lambda x: (not str(x).endswith("_00.csv"), str(x))
    )
    for i, file in enumerate(files):
        # for i, file in enumerate(folder_dir.iterdir()):
        # only read the raw files
        if "raw" not in file.name:
            continue

        # Read csv file
        df = read_csv_with_locale(file)
        # compute rounded vw
        vw_unique_round = int(round(df["vw"].unique()[0], 0))
        print(f"\n---------------------")
        print(f"File: {file.name}, \nvw: {vw_unique_round}")

        # If zz correcting for the sideslip
        if is_zigzag:
            df["sideslip"] -= 1.8

        # Storing the raw values
        df["F_X_raw"] = df["F_X"]
        df["F_Y_raw"] = df["F_Y"]
        df["F_Z_raw"] = df["F_Z"]
        df["M_X_raw"] = df["M_X"]
        df["M_Y_raw"] = df["M_Y"]
        df["M_Z_raw"] = df["M_Z"]

        # 2. compute Average Zero-Run values
        if df["vw"].unique()[0] < 1.5:
            print(f'i: {i}, vw: {df["vw"].unique()}')
            # print(f'i: {i}, vw: {df["vw"].unique()}')
            # Computing the average measured zero-run values
            zero_F_X = df["F_X"].mean()
            zero_F_Y = df["F_Y"].mean()
            zero_F_Z = df["F_Z"].mean()
            zero_M_X = df["M_X"].mean()
            zero_M_Y = df["M_Y"].mean()
            zero_M_Z = df["M_Z"].mean()

        ## Skipping vw = 5 for zigzag
        elif vw_unique_round == 5 and is_zigzag:
            print(f"Not processing because zigzag and vw: {vw_unique_round}")
            continue
        else:

            # TODO: This can be simplfied and taken from the filename
            # Correcting the angle of attack, rounding to 1 decimal (like the folder name)
            df["aoa_kite"] = round(df["aoa"] - delta_aoa_rod_to_alpha, 1)

            # And add dynamic viscosity and Reynolds number
            df = add_dyn_visc_and_reynolds(
                df, delta_celsius_kelvin, mu_0, T_0, C_suth, c_ref
            )

            # 1. Substracting the zero-run values
            df["F_X"] = df["F_X"] - zero_F_X
            df["F_Y"] = df["F_Y"] - zero_F_Y
            df["F_Z"] = df["F_Z"] - zero_F_Z
            df["M_X"] = df["M_X"] - zero_M_X
            df["M_Y"] = df["M_Y"] - zero_M_Y
            df["M_Z"] = df["M_Z"] - zero_M_Z

            ###### Processing the data ######
            # 1. Translate Moments from the load balance origin to the kite cg
            #       important to do this before non-dimensionalizing as ur multplying by distance
            # 2. Then rotate the axis to the new coordinate system [x, y, z] = [x, -y, -z]
            # 3. Correct for sideslip, where beta = -beta
            #       and a rotation is made to have cd not align with chord, but with incoming v_wind

            df = translate_from_origin_to_cg(
                df,
                x_hinge,
                z_hinge,
                l_cg,
                alpha_cg_delta_with_rod,
            )
            df = rotate_axis_at_origin(df)
            df = correcting_for_sideslip(df)

            #### Non-dimensionalize (F_X --> CF_X)
            df = nondimensionalize(df, S_ref, c_ref)

            # 7. Calculate signal-to-noise ratio
            ## comparing RAW --to-- (RAW - zero-run)
            df["SNR_CF_X"] = df["CF_X"] / df["CF_X_raw"]
            df["SNR_CF_Y"] = df["CF_Y"] / df["CF_Y_raw"]
            df["SNR_CF_Z"] = df["CF_Z"] / df["CF_Z_raw"]
            df["SNR_CM_X"] = df["CM_X"] / df["CM_X_raw"]
            df["SNR_CM_Y"] = df["CM_Y"] / df["CM_Y_raw"]
            df["SNR_CM_Z"] = df["CM_Z"] / df["CM_Z_raw"]

            # ## comparing with support measured --to-- raw, only difference is zero-run
            # df["SNR_CF_X"] = df["F_X"] / df["F_X_raw"]
            # df["SNR_CF_Y"] = df["F_Y"] / df["F_Y_raw"]
            # df["SNR_CF_Z"] = df["F_Z"] / df["F_Z_raw"]
            # df["SNR_CM_X"] = df["M_X"] / df["M_X_raw"]
            # df["SNR_CM_Y"] = df["M_Y"] / df["M_Y_raw"]
            # df["SNR_CM_Z"] = df["M_Z"] / df["M_Z_raw"]

            ### Creating lift,drag,etc.
            df["C_D"] = df["CF_X"]
            df["C_S"] = df["CF_Y"]
            df["C_L"] = df["CF_Z"]
            df["C_roll"] = df["CM_X"]
            df["C_pitch"] = df["CM_Y"]
            df["C_yaw"] = df["CM_Z"]

            if is_kite:
                # 5. Substracting support structure aerodynamic coefficients ("C_D" in "C_D" out)
                df = substract_support_structure_aero_coefficients(
                    df, support_struc_aero_interp_coeffs_path
                )
                # # 6. Apply wind-tunnel corrections
                df = apply_wind_tunnel_corrections(df)

            # Dropping columns that are no longer needed
            columns_to_drop = [
                "aoa",
                "F_X",
                "F_Y",
                "F_Z",
                "M_X",
                "M_Y",
                "M_Z",
                "F_X_raw",
                "F_Y_raw",
                "F_Z_raw",
                "M_X_raw",
                "M_Y_raw",
                "M_Z_raw",
                "CF_X",
                "CF_Y",
                "CF_Z",
                "CM_X",
                "CM_Y",
                "CM_Z",
                "CF_X_raw",
                "CF_Y_raw",
                "CF_Z_raw",
                "CM_X_raw",
                "CM_Y_raw",
                "CM_Z_raw",
                # "Filename",
                "Date",
                "Dpa",
                "Pressure",
                "Temp",
                "Density",
            ]
            for col in columns_to_drop:
                if col not in df.columns:
                    continue
                df.drop(columns=[col], inplace=True)

            # Save the processed data to a csv file
            # file_name_without_raw = df["Filename"].unique()[0]
            new_file_name = f"vw_{vw_unique_round:.0f}"
            if "ZZ" in file.name:
                new_file_name = f"ZZ_{new_file_name}"
            print(f"saving file: {new_file_name}")
            df.to_csv(folder_dir / f"{new_file_name}.csv", index=False)
            # print(f"END columns: {df.columns}")
