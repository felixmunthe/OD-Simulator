# -*- coding: utf-8 -*-
"""
Outlier Detection Models

Author: Munthe, Felix A.
Created on Friday, 23 June 2023
"""

import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import Pseudopressure_Conversion
from svg_to_emf import convert_svg_to_emf

# ----- Import Production Data -----
def read_production(file_path):

    # Step 1: Open the text file
    with open(file_path, "r", encoding = "utf-8") as file:
        lines = file.readlines()

    # Step 2: Process the data and create an array
    time = []
    downhole_rate = []
    
    # Process each subsequent line and append the values
    for line in lines[2:]:
        values = line.split()
        if len(values) >= 2:
            time.append(float(values[0]))
            downhole_rate.append(float(values[1]))
    
    return time, downhole_rate

# ----- Adding Noise Single Time Interval -----
def add_noise(coordinates, time_data, time_interval, noise_fraction):

    # Step 1: Find indices of coordinates within the given time range
    indices = [i for i, time in enumerate(time_data) if time_interval[0] < time <= time_interval[1]]
    
    # Step 2: Select coordinates within the time interval
    selected_coordinates = [coordinates[i] for i in indices]

    # Step 3: Extract pressure values from the selected coordinates
    selected_pressure = [coord[1] for coord in selected_coordinates]

    # Step 4: Calculate mean and standard deviation
    mean = np.mean(selected_pressure)
    std_dev = np.std(selected_pressure)

    # Step 5: Choose percentage of selected coordinates randomly
    num_coordinates_change = int(math.ceil(len(selected_coordinates) * noise_fraction))
    indices_change = random.sample(range(len(selected_coordinates)), num_coordinates_change)

    # Step 6: Apply Additive White Gaussian Noise (AWGN) to the selected coordinates
    noisy_coordinates = coordinates.copy()
    for i, coord in enumerate(selected_coordinates):
        if i in indices_change:
            time, pressure = coord
            noise_multiplier = random.uniform(-3, 3)
            while noise_multiplier == 0:
                noise_multiplier = random.uniform(-3, 3)
            noise_pressure = noise_multiplier * std_dev

            # Repeat process until if positive value doesn't obtained
            while True:
                noisy_coord = (time, pressure + noise_pressure)
                if noisy_coord[1] > 0:
                    break
                else:
                    noise_multiplier = random.uniform(-3, 3)
                    while noise_multiplier == 0:
                        noise_multiplier = random.uniform(-3, 3)
                    noise_pressure = noise_multiplier * std_dev
            
            noisy_coordinates[indices[i]] = noisy_coord

    return noisy_coordinates

# ----- Export Noisy Data -----
def export_noisy(data, filename, folder_path, header = None):

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    file_path = os.path.join(folder_path, filename)
    with open(file_path, "w") as file:
        if header is not None:
            file.write(header + "\n")
        for item in data:
            file.write(f"{item[0]}\t{item[1]}\n")
    return

# ----- Main Execution -----
def main():

    global time_prod, RNP_coordinates, num_noisy_generate
    
    # Production data file path
    downhole_file_path = r"C:\Users\felix\OneDrive\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Outlier Detection Paper\Outlier Detection Simulator\Synthetic Model\downhole_gas_rate.txt"
    surface_file_path = r"C:\Users\felix\OneDrive\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Outlier Detection Paper\Outlier Detection Simulator\Synthetic Model\surface_gas_rate.txt"
    volume_file_path = r"C:\Users\felix\OneDrive\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Outlier Detection Paper\Outlier Detection Simulator\Synthetic Model\surface_gas_volume.txt"

    # Define time and rate raw data
    time, downhole_rate = read_production(downhole_file_path) # hr, Mcf/d
    time, surface_rate = read_production(surface_file_path) # hr, Mscf/d
    time, volume = read_production(volume_file_path) # hr, scf
    time_prod = time[1:] # hr
    time_prod = np.array(time_prod) / 24 # days
    volume = np.array(volume) * 0.001 # Mscf
    tmb = [volume[i] / surface_rate[i] for i in range (1, len(time))] # days

    # Generate RNP coordinates
    RNP_data = [Pseudopressure_Conversion.delta_pseudopressure[i] / downhole_rate[i] for i in range(1, len(time))]
    RNP_coordinates = list(zip(time_prod, RNP_data))

    # Define time interval to add noise to data
    transient = (0, 4000)
    transition = (4000, 40000)
    bdf = (40000, 73000)
    print(len)
    # Define additional time intervals
    transient_transition = (transient, transition)
    transient_bdf = (transient, bdf)
    transition_bdf = (transition, bdf)
    all = (transient, transition, bdf)

    # Create a list of the time intervals
    time_intervals = [transient, transition, bdf, transient_transition, transient_bdf, transition_bdf, all]

    # Number of noisy data generated
    num_noisy_generate = 10
    noise_fractions = [0.25, 0.5, 0.75]

    # Define base folder path for exporting text files
    base_folder_path = r"C:\Users\felix\OneDrive\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Outlier Detection Paper\Outlier Detection Simulator\Noisy RNP Model"
    
    # Create a dictionary to folder
    interval_names = {
        transient: "Transient",
        transition: "Transition",
        bdf: "BDF",
        transient_transition: "Transient-Transition",
        transient_bdf: "Transient-BDF",
        transition_bdf: "Transition-BDF",
        all: "All"
    }
    
    # # Loop over the time intervals
    # for interval_index, time_interval in enumerate(time_intervals):

    #     # Create folder for specific time interval
    #     interval_folder = interval_names.get(time_interval)
    #     interval_path = os.path.join(base_folder_path, interval_folder)

    #     if not os.path.exists(interval_path):
    #         os.makedirs(interval_path)

    #     # Loop over the noise fractions
    #     for noise_fraction in noise_fractions:

    #         # Create folder for specific noise fraction
    #         noise_folder = f"{int(noise_fraction * 100)}%"
    #         noise_path = os.path.join(interval_path, noise_folder)
        
    #         if not os.path.exists(noise_path):
    #             os.makedirs(noise_path)
    
    #         # Add noise to the RNP coordinates based on time inverval
    #         for i in range (num_noisy_generate):
    #             noisy_data = []
                
    #             if time_interval == transient or time_interval == transition or time_interval == bdf:
    #                 noisy_coordinates = add_noise(RNP_coordinates, time_prod, time_interval, noise_fraction)
    #             elif time_interval == transient_transition or time_interval == transient_bdf or time_interval == transition_bdf:
    #                 temp_coordinates = add_noise(RNP_coordinates, time_prod, time_interval[0], noise_fraction)
    #                 noisy_coordinates = add_noise(temp_coordinates, time_prod, time_interval[1], noise_fraction)
    #             elif time_interval == all:
    #                 temp_coordinates = add_noise(RNP_coordinates, time_prod, time_interval[0], noise_fraction)
    #                 temps_coordinates = add_noise(temp_coordinates, time_prod, time_interval[1], noise_fraction)
    #                 noisy_coordinates = add_noise(temps_coordinates, time_prod, time_interval[2], noise_fraction)
            
    #             noisy_data.append(noisy_coordinates)
        
    #             # Export the noisy data to text file
    #             filename = f"noisy_data_{i + 1}.txt"
    #             header = "t(days)\tRNP(psia2/cp-D/Mscf)"
    #             export_noisy(noisy_coordinates, filename, noise_path, header)
    
    # Plot RNP vs time
    plt.figure()
    plt.plot(time_prod, RNP_data, 'o', label = "constant bottomhole pressure (actual time)")
    plt.plot(tmb, RNP_data, '^', label = "constant rate (material-balance time)")
    plt.xlabel("$t$, day", fontsize = 20)
    plt.ylabel("$\Delta$m(p)/$q_{g}$, $psia^{2}$/cp$\cdot$d/Mscf", fontsize = 20)
    plt.xscale("log")
    plt.yscale("log")
    #plt.title("Rate-normalized Pseudo-pressure (RNP) vs. Time", fontsize = 24)
    plt.minorticks_on()
    plt.tick_params(axis = 'both', which = 'major', labelsize = 18)
    plt.grid(True, which = 'major', linestyle = '-', linewidth = 1)
    plt.grid(True, which = 'minor', linestyle = ':', linewidth = 0.5)
    plt.legend(fontsize = 20)
    plt.show()

    svg_path = r"C:\Users\felix\OneDrive\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Outlier Detection Paper\Outlier Detection Simulator\Figures\Fig_9.svg"
    emf_path = r"C:\Users\felix\OneDrive\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Outlier Detection Paper\Outlier Detection Simulator\Figures\Fig_9.emf"
    inkscape_path = r"C:\Program Files\Inkscape\bin\inkscape.exe"

    convert_svg_to_emf(svg_path, emf_path, inkscape_path=inkscape_path)

    # Plot rate vs tmb
    # plt.figure()
    # plt.plot(tmb, downhole_rate[1:], ".")
    # plt.xlabel("$t_{mb}$, hr")
    # plt.ylabel("$q_{g}$, Mscf/d")
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.title("Rate vs. Material Balance Time (MBT)")
    # plt.minorticks_on()
    # plt.grid(True, which = 'major', linestyle = '-', linewidth = 1)
    # plt.grid(True, which = 'minor', linestyle = ':', linewidth = 0.5)
    # plt.show()

    return

main()