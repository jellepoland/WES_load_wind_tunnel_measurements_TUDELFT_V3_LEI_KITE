def calculate_wind_tunnel_corrections(
    A=0.462,  # model area in m²
    C=7.47,  # tunnel jet-exhaust area in m²
    b=1.287,  # geometric span (width) in m
    bv_to_b=0.785,  # ratio of vortex span to geometric span
    delta=-0.126,  # empirically determined factor
    tau_2=0.054,  # streamline curvature factor
    tau_2_2=0.108,  # streamline curvature factor with half chord length
    tau_2_s=0.028,  # sideways streamline curvature factor
    tau_2_2_s=0.056,  # sideways streamline curvature factor with half chord length
    dCL_dalpha=0.1,  # derivative of lift coefficient w.r.t. angle of attack
    dCS_dbeta=0.01,  # derivative of side force coefficient w.r.t. sideslip angle
    C_LW=None,  # wing lift coefficient (optional)
    C_S=None,  # side force coefficient (optional)
):
    """
    Calculate wind tunnel corrections based on the equations from Barlow (1999).

    Parameters are described above with their default values from the document.
    If C_LW and C_S are provided, the function will calculate the actual corrections.
    If not provided, the function will return the coefficients that multiply with C_LW or C_S.

    Returns a dictionary with all calculated values.
    """
    # Create a dictionary to store all values
    results = {}

    # Store input parameters
    results["inputs"] = {
        "A": A,
        "C": C,
        "b": b,
        "bv_to_b": bv_to_b,
        "delta": delta,
        "tau_2": tau_2,
        "tau_2_2": tau_2_2,
        "tau_2_s": tau_2_s,
        "tau_2_2_s": tau_2_2_s,
        "dCL_dalpha": dCL_dalpha,
        "dCS_dbeta": dCS_dbeta,
        "C_LW": C_LW,
        "C_S": C_S,
    }

    # ===== Calculate effective vortex span b_e =====
    b_e = (b / 2) * (1 + bv_to_b)
    results["b_e"] = b_e

    # ===== Downwash and Streamline Curvature Corrections =====

    # Downwash angle correction (in radians)
    delta_alpha_base = delta * (A / C)
    results["delta_alpha_base"] = delta_alpha_base

    # Streamline curvature angle correction (in radians)
    delta_alpha_sc_base = tau_2 * delta_alpha_base
    results["delta_alpha_sc_base"] = delta_alpha_sc_base

    # Total angle correction (in radians)
    delta_alpha_t_rad_base = delta_alpha_base + delta_alpha_sc_base
    results["delta_alpha_t_rad_base"] = delta_alpha_t_rad_base

    # Total angle correction (in degrees)
    delta_alpha_t_deg_base = delta_alpha_t_rad_base * (180 / 3.14159)
    results["delta_alpha_t_deg_base"] = delta_alpha_t_deg_base

    # Drag coefficient correction
    delta_CD_base = delta * (A / C)
    results["delta_CD_base"] = delta_CD_base

    # Lift coefficient correction
    delta_CL_base = -delta_alpha_sc_base * dCL_dalpha
    results["delta_CL_base"] = delta_CL_base

    # Pitching moment coefficient correction
    delta_alpha_sc_2_base = tau_2_2 * delta_alpha_base
    results["delta_alpha_sc_2_base"] = delta_alpha_sc_2_base
    delta_CM_y_base = 0.125 * delta_alpha_sc_2_base * dCL_dalpha
    results["delta_CM_y_base"] = delta_CM_y_base

    # ===== Side Force Corrections =====

    # Sideslip angle correction (in radians)
    delta_beta_base = delta * (A / C)
    results["delta_beta_base"] = delta_beta_base

    # Streamline curvature for sideslip (in radians)
    delta_beta_sc_base = tau_2_s * delta_beta_base
    results["delta_beta_sc_base"] = delta_beta_sc_base

    # Total sideslip angle correction (in radians)
    delta_beta_t_rad_base = delta_beta_base + delta_beta_sc_base
    results["delta_beta_t_rad_base"] = delta_beta_t_rad_base

    # Total sideslip angle correction (in degrees)
    delta_beta_t_deg_base = delta_beta_t_rad_base * (180 / 3.14159)
    results["delta_beta_t_deg_base"] = delta_beta_t_deg_base

    # Drag coefficient correction from side force
    delta_CD_side_base = delta * (A / C)
    results["delta_CD_side_base"] = delta_CD_side_base

    # Side force coefficient correction
    delta_CS_base = -delta_beta_sc_base * dCS_dbeta
    results["delta_CS_base"] = delta_CS_base

    # Yawing moment coefficient correction
    delta_beta_sc_2_base = tau_2_2_s * delta_beta_base
    results["delta_beta_sc_2_base"] = delta_beta_sc_2_base
    delta_CM_z_base = 0.125 * delta_beta_sc_2_base * dCS_dbeta
    results["delta_CM_z_base"] = delta_CM_z_base

    # ===== Actual Corrections (if C_LW and C_S are provided) =====
    if C_LW is not None:
        results["delta_alpha_t_deg"] = delta_alpha_t_deg_base * C_LW
        results["delta_CD_lift"] = delta_CD_base * (C_LW**2)
        results["delta_CL"] = delta_CL_base * C_LW
        results["delta_CM_y"] = delta_CM_y_base * C_LW

    if C_S is not None:
        results["delta_beta_t_deg"] = delta_beta_t_deg_base * C_S
        results["delta_CD_side"] = delta_CD_side_base * (C_S**2)
        results["delta_CS"] = delta_CS_base * C_S
        results["delta_CM_z"] = delta_CM_z_base * C_S

    return results


def print_wind_tunnel_corrections(results):
    """Print the results in a formatted way."""
    print("\n" + "=" * 60)
    print(" " * 20 + "WIND TUNNEL CORRECTIONS")
    print("=" * 60)

    # Print input parameters
    print("\nINPUT PARAMETERS:")
    print("-" * 60)
    inputs = results["inputs"]
    for key, value in inputs.items():
        if value is not None:
            print(f"{key:12} = {value:.6f}")

    # Print effective vortex span
    print("\nEFFECTIVE VORTEX SPAN:")
    print("-" * 60)
    print(f"b_e = {results['b_e']:.6f} m")

    # Print intermediate calculations for downwash and streamline curvature
    print("\nDOWNWASH AND STREAMLINE CURVATURE CALCULATIONS:")
    print("-" * 60)
    print(f"δ(A/C)                           = {results['delta_alpha_base']:.6f}")
    print(f"τ₂ × δ(A/C)                      = {results['delta_alpha_sc_base']:.6f}")
    print(
        f"Total Δα_t (rad) base coefficient = {results['delta_alpha_t_rad_base']:.6f}"
    )
    print(
        f"Total Δα_t (deg) base coefficient = {results['delta_alpha_t_deg_base']:.6f}"
    )
    print(f"ΔC_D base coefficient             = {results['delta_CD_base']:.6f}")
    print(f"ΔC_L base coefficient             = {results['delta_CL_base']:.6f}")
    print(f"τ₂₍₂₎ × δ(A/C)                    = {results['delta_alpha_sc_2_base']:.6f}")
    print(f"ΔC_M,y base coefficient           = {results['delta_CM_y_base']:.6f}")

    # Print intermediate calculations for side force
    print("\nSIDE FORCE CALCULATIONS:")
    print("-" * 60)
    print(f"δ(A/C) for sideslip               = {results['delta_beta_base']:.6f}")
    print(f"τ₂,ₛ × δ(A/C)                     = {results['delta_beta_sc_base']:.6f}")
    print(f"Total Δβ_t (rad) base coefficient = {results['delta_beta_t_rad_base']:.6f}")
    print(f"Total Δβ_t (deg) base coefficient = {results['delta_beta_t_deg_base']:.6f}")
    print(f"ΔC_D (side) base coefficient      = {results['delta_CD_side_base']:.6f}")
    print(f"ΔC_S base coefficient             = {results['delta_CS_base']:.6f}")
    print(f"τ₂₍₂₎,ₛ × δ(A/C)                  = {results['delta_beta_sc_2_base']:.6f}")
    print(f"ΔC_M,z base coefficient           = {results['delta_CM_z_base']:.6f}")

    # Print actual corrections if C_LW or C_S were provided
    if inputs["C_LW"] is not None:
        print("\nACTUAL LIFT-BASED CORRECTIONS:")
        print("-" * 60)
        print(f"Δα_t (deg)                      = {results['delta_alpha_t_deg']:.6f}")
        print(f"ΔC_D (from lift)                = {results['delta_CD_lift']:.6f}")
        print(f"ΔC_L                            = {results['delta_CL']:.6f}")
        print(f"ΔC_M,y                          = {results['delta_CM_y']:.6f}")
    else:
        print("\nLIFT-BASED CORRECTIONS (multiply by C_LW or C_LW²):")
        print("-" * 60)
        print(f"Δα_t (deg) = {results['delta_alpha_t_deg_base']:.6f} × C_LW")
        print(f"ΔC_D       = {results['delta_CD_base']:.6f} × C_LW²")
        print(f"ΔC_L       = {results['delta_CL_base']:.6f} × C_LW")
        print(f"ΔC_M,y     = {results['delta_CM_y_base']:.6f} × C_LW")

    if inputs["C_S"] is not None:
        print("\nACTUAL SIDE FORCE-BASED CORRECTIONS:")
        print("-" * 60)
        print(f"Δβ_t (deg)                      = {results['delta_beta_t_deg']:.6f}")
        print(f"ΔC_D (from side force)          = {results['delta_CD_side']:.6f}")
        print(f"ΔC_S                            = {results['delta_CS']:.6f}")
        print(f"ΔC_M,z                          = {results['delta_CM_z']:.6f}")
    else:
        print("\nSIDE FORCE-BASED CORRECTIONS (multiply by C_S or C_S²):")
        print("-" * 60)
        print(f"Δβ_t (deg) = {results['delta_beta_t_deg_base']:.6f} × C_S")
        print(f"ΔC_D       = {results['delta_CD_side_base']:.6f} × C_S²")
        print(f"ΔC_S       = {results['delta_CS_base']:.6f} × C_S")
        print(f"ΔC_M,z     = {results['delta_CM_z_base']:.6f} × C_S")


# Example usage
if __name__ == "__main__":
    # Example 1: Using default values from the document, without C_LW and C_S
    print("\nExample 1: Using default values, coefficients only")
    results = calculate_wind_tunnel_corrections()
    print_wind_tunnel_corrections(results)

    # # Example 2: Using default values with specific C_LW and C_S
    # print("\nExample 2: Using default values with C_LW = 0.5 and C_S = 0.3")
    # results = calculate_wind_tunnel_corrections(C_LW=0.5, C_S=0.3)
    # print_wind_tunnel_corrections(results)

    # # Example 3: Changing some parameters
    # print("\nExample 3: Modified parameters")
    # results = calculate_wind_tunnel_corrections(
    #     A=0.5,  # Changed model area
    #     C=8.0,  # Changed tunnel area
    #     delta=-0.15,  # Changed empirical factor
    #     tau_2=0.06,  # Changed streamline curvature factor
    #     C_LW=0.7,  # With wing lift coefficient
    #     C_S=0.4,  # With side force coefficient
    # )
    # print_wind_tunnel_corrections(results)
