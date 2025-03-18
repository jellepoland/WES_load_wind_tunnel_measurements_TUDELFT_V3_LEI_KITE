from pathlib import Path
import pandas as pd
import numpy as np
from load_balance_analysis.functions_utils import project_dir


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


# open csv
df = pd.read_csv(Path(project_dir) / "data" / "processed_labbook.csv")

# take the vw column and round each value to 0 decimal places
df["vw"] = df["vw"].round(0)

print(df["vw"])

df_5 = df.loc[df["vw"] == 5]
df_10 = df.loc[df["vw"] == 10]
df_15 = df.loc[df["vw"] == 15]
df_20 = df.loc[df["vw"] == 20]
df_25 = df.loc[df["vw"] == 25]

df_list = [df_5, df_10, df_15, df_20, df_25]
u_arr = np.array([5, 10, 15, 20, 25])

for df_i, u_i in zip(df_list, u_arr):
    temp_i = df_i["Temp"].mean()
    rho_i = df_i["Density"].mean()
    nu_i = kinematic_viscosity_air(T_deg=temp_i, rho=rho_i)
    print(
        f"\n u: {u_i} m/s, T: {temp_i} degC, rho: {rho_i} kg/m^3, nu: {nu_i:.3e} m^2/s"
    )

    re = (u_i * 0.395) / nu_i
    print(f"Re: {re:.3e}")
    # print(df_i["Temp"])
    temp = df_i["Temp"].values
    # print(f"Temp: {temp}")
    print(np.mean(temp))


# # compute average values of all columns an dprint
# print("rho:", df["Density"].mean())
# print("T:", df["Temp"].mean())
# print(
#     f'nu: {kinematic_viscosity_air(T_deg=df["Temp"].mean(), rho=df["Density"].mean()):.3e} m^2/s'
# )
# print(f"nu: {kinematic_viscosity_air(T_deg=25.6, rho=1.17):.3e} m^2/s")

# u_arr = np.array([5, 10, 15, 20, 25])
# for u in u_arr:
#     re = u * 0.395 / (kinematic_viscosity_air(T_deg=25.6, rho=1.17))
#     print(f"u: {u} m/s, Re: {re:.2e}")
