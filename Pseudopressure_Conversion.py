# -*- coding: utf-8 -*-
"""
Pseudopressure Converter

Author: Munthe, Felix A.
Created on Friday, 23 June 2023
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.interpolate import interp1d

# ----- Import PVT Data -----
def read_gas_properties (file_path):

    # Step 1: Open the text file
    with open(file_path, "r", encoding = "utf-8") as file:
        
        # Step 2: Read the file contents
        lines = file.readlines()

    # Step 3: Process data and create arrays
    pvt_pressure = []
    pvt_Z = []
    pvt_mug = []
    pvt_cg = []
    
    # Step 4: Process each subsequent line and append the values
    for line in lines[2:]:
        values = line.split()
        pvt_pressure.append(float(values[0]))
        pvt_Z.append(float(values[1]))
        pvt_mug.append(float(values[2]))
        pvt_cg.append(float(values[3]))
    
    return pvt_pressure, pvt_Z, pvt_mug, pvt_cg

# ----- Import Bottomhole Pressure Data -----
def read_production (file_path):

    # Step 1: Open the text file
    with open(file_path, "r", encoding = "utf-8") as file:
    
        # Step 2: Read the file contents
        lines = file.readlines()

    # Step 3: Process the data and create an array
    time = []
    bhp = []
    
    # Process each subsequent line and append the values
    for line in lines[2:]:
        values = line.split()
        if len(values) >= 2:
            time.append(float(values[0]))
            bhp.append(float(values[1]))
    
    return time, bhp

# ----- Import Reservoir Pressure Data -----
def read_reservoir (file_path):

    # Step 1: Open the text file
    with open(file_path, "r", encoding = "utf-8") as file:
    
        # Step 2: Read the file contents
        lines = file.readlines()

    # Step 3: Process the data and create an array
    res_pressure = []
    
    # Process each subsequent line and append the values
    for line in lines[2:]:
        values = line.split()
        if len(values) >= 2:
            res_pressure.append(float(values[1]))
    
    return res_pressure

# ----- Interpolation -----
def interpolate_data (ref_point, ref_values, target_point):
    
    # Step 1: Create interpolation functions for pseudopressure
    interp_calculation = interp1d(ref_point, ref_values, kind = 'linear')

    # Step 2: Interpolate pseudopressure at target point
    interpolated_point = interp_calculation(target_point)

    return interpolated_point

# ----- Pseudopressure Calculation -----
def calculate_pseudopressure(pressure, mug, Z):

    pseudopressure = []

    for p, mu, z in zip(pressure, mug, Z):
        def integrand(p_val):
            return p_val / (mu * z)
        
        result, _ = integrate.quad(integrand, 0, p)
        pseudopressure.append(2 * result)
    
    return pseudopressure

# ----- Pseudotime Calculation -----
def calculate_pseudotime(time, res_pressure, pvt_pressure, pvt_mug, ct):
    time = np.asarray(time, dtype = float)
    res_pressure = np.asarray(res_pressure, dtype = float)

    mug_i = interpolate_data(pvt_pressure, pvt_mug, res_pressure[0])
    ct_i = interpolate_data(pvt_pressure, ct, res_pressure[0])

    mug_res = interpolate_data(pvt_pressure, pvt_mug, res_pressure)
    ct_res = interpolate_data(pvt_pressure, ct, res_pressure)

    pseudotime = np.zeros(len(time), dtype = float)

    for n in range(1, len(time)):
        dt = time[n] - time[n - 1]
        if dt < 0:
            raise ValueError("Time array must be increasing.")

        inv_mu_ct_avg = 0.5 * (
            1.0 / (mug_res[n] * ct_res[n]) +
            1.0 / (mug_res[n - 1] * ct_res[n - 1])
        )

        pseudotime[n] = pseudotime[n - 1] + (mug_i * ct_i) * dt * inv_mu_ct_avg

    return pseudotime

def calculate_linear_superposition_pseudotime(pseudotime, rate):
    pseudotime = np.asarray(pseudotime, dtype = float)
    rate = np.asarray(rate, dtype = float)

    if len(pseudotime) != len(rate):
        raise ValueError("pseudotime and rate must have the same length.")

    # Extend arrays so q_0 = 0 and tca_0 = 0
    q_ext = np.concatenate(([0.0], rate))
    tca_ext = np.concatenate(([0.0], pseudotime))

    pseudotime_linear = np.zeros(len(rate), dtype = float)

    for n in range(1, len(q_ext)):
        qn = q_ext[n]

        if qn <= 0:
            pseudotime_linear[n - 1] = np.nan
            continue

        summation = 0.0
        for j in range(1, n + 1):
            dq = q_ext[j] - q_ext[j - 1]
            dtca = tca_ext[n] - tca_ext[j - 1]

            if dtca < 0:
                dtca = 0.0

            summation += (dq / qn) * np.sqrt(dtca)

        pseudotime_linear[n - 1] = summation ** 2

    return pseudotime_linear

# ----- Main Code -----
def main():
    global downhole_rate
    gas_properties_file_path = r"C:\Users\felix\OneDrive\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Outlier Detection Paper\Outlier Detection Simulator\Synthetic Model\gas_properties.txt"
    pvt_pressure, pvt_Z, pvt_mug, pvt_cg = read_gas_properties(gas_properties_file_path) # psia, , cp, 1/psi

    downhole_file_path = r"C:\Users\felix\OneDrive\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Outlier Detection Paper\Outlier Detection Simulator\Synthetic Model\downhole_gas_rate.txt"
    time, downhole_rate = read_production(downhole_file_path) # hrs., Mcf/d

    bhp_file_path = r"C:\Users\felix\OneDrive\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Outlier Detection Paper\Outlier Detection Simulator\Synthetic Model\bhp.txt"
    time, bhp = read_production(bhp_file_path) # hrs., psia
    time = np.array(time) / 24 # days
    
    reservoir_file_path = r"C:\Users\felix\OneDrive\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Outlier Detection Paper\Outlier Detection Simulator\Synthetic Model\res_pressure.txt"
    res_pressure = read_reservoir(reservoir_file_path) # psia
    
    pvt_pseudopressure = calculate_pseudopressure(pvt_pressure, pvt_mug, pvt_Z)
    
    # # Plot pseudopressure vs pressure
    # plt.figure()
    # plt.plot(pvt_pressure, pvt_pseudopressure, "-")
    # plt.xlabel("p, psia")
    # plt.ylabel("m(p), $psia^{2}$/cp")
    # plt.title("Pseudopressure vs. Pressure")
    # plt.minorticks_on()
    # plt.grid(True, which = 'major', linestyle = '-', linewidth = 1)
    # plt.grid(True, which = 'minor', linestyle = ':', linewidth = 0.5)
    # plt.ylim(bottom=0)
    # plt.xlim(left=0)
    # plt.show()

    global cti, pseudotime, pseudotime_linear
    cf = 3e-6 # 1/psi
    Soi = 0
    Sgi = 0.9
    Swi = 0.1
    cw = 2.8e-6
    ct = [cf + pvt_cg[i] * Sgi + cw * Swi for i in range(len(pvt_cg))]

    downhole_rate = np.array(downhole_rate, dtype = float)

    pseudotime = calculate_pseudotime(time, res_pressure, pvt_pressure, pvt_mug, ct)
    pseudotime_linear = calculate_linear_superposition_pseudotime(pseudotime, downhole_rate)

    res_pressure_init = res_pressure[0]
    cti = interpolate_data(pvt_pressure, ct, res_pressure_init)

    bhp_pseudopressure = interpolate_data(pvt_pressure, pvt_pseudopressure, bhp)
    bhp_pseudopressure = np.array(bhp_pseudopressure)

    res_pressure_pseudopressure  = []
    for i in range (len(pseudotime)):
        res_pressure_pseudopressure.append(interpolate_data(pvt_pressure, pvt_pseudopressure, res_pressure_init))
    res_pressure_pseudopressure = np.array(res_pressure_pseudopressure)
    
    global delta_pseudopressure
    delta_pseudopressure = [res_pp - bhp_pp for res_pp, bhp_pp in zip(res_pressure_pseudopressure, bhp_pseudopressure)]
    delta_pseudopressure = np.array(delta_pseudopressure)

    # # Plot delta pseudopressure vs time
    # plt.figure()
    # plt.plot(time, res_pressure_pseudopressure, "-", label = "Reservoir")
    # plt.plot(time, bhp_pseudopressure, "-", label = "Downhole")
    # plt.plot(time, delta_pseudopressure, "-", label = "Delta")
    # plt.xlabel("t, hr")
    # plt.ylabel("m(p), $psia^{2}$/cp")
    # plt.title("Pseudopressure vs. Time")
    # plt.minorticks_on()
    # plt.grid(True, which = 'major', linestyle = '-', linewidth = 1)
    # plt.grid(True, which = 'minor', linestyle = ':', linewidth = 0.5)
    # plt.ylim(bottom=0)
    # plt.xlim(left=0)
    # plt.legend()
    # plt.show()
    
    # Create a new folder for the results
    output_folder = "Pseudopressure Conversion"
    os.makedirs(output_folder, exist_ok = True)

    # Export pseudopressure arrays to text file
    pseudopressure_path = os.path.join(output_folder, "pvt_pseudopressure.txt")
    np.savetxt(pseudopressure_path, np.column_stack((pvt_pressure, pvt_pseudopressure)), delimiter = "\t", header="PVT_p\tPVT_m(p)", fmt = "%.6f", comments = "")
    delta_pseudopressure_path = os.path.join(output_folder, "delta_pseudopressure.txt")
    np.savetxt(delta_pseudopressure_path, np.column_stack((time, res_pressure, bhp, res_pressure_pseudopressure, bhp_pseudopressure, delta_pseudopressure)), delimiter = "\t", header="t\tReservoir_p\tbhp\tReservoir_m(p)\tDownhole_m(p)\tDelta_m(p)", fmt = "%.6f", comments = "")

    return

main()