# -*- coding: utf-8 -*-
"""
Outlier Detection Models

Author: Munthe, Felix A.
Created on Friday, 23 June 2023
"""

import os
import ast
import json
import csv
import time
import Pseudopressure_Conversion
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, accuracy_score, recall_score, precision_score, f1_score
from OD_Models import (
    OD_KDE, OD_GMM, OD_HBOS, OD_COPOD, OD_ECOD,
    OD_LMDD, OD_MCD, OD_QMCD, OD_PCA, OD_KPCA,
    OD_LOF, OD_COF, OD_KNN, OD_ABOD, OD_SOS,
    OD_FB, OD_IF, OD_LODA, OD_INNE, OD_DIF
)
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

# ----- Import Noisy Data -----
def read_noisy (file_path):

    # Step 1: Open the text file
    with open (file_path, "r", encoding = "utf-8") as file:
        lines = file.readlines()

    # Step 2: Process the data and create an array
    noisy_time = []
    noisy_RNP = []
    
    # Process each subsequent line and append the values
    for line in lines[1:]:
        values = line.split()
        if len(values) >= 2:
            noisy_time.append(float(values[0]))
            noisy_RNP.append(float(values[1]))
    
    return noisy_time, noisy_RNP

# ----- Assign Label Data -----
def data_label(true_data, predicted_data):

    # Initialize label array with all ones (default value)
    label = np.ones(len(predicted_data), dtype = int)

    # Check if each element of B exists in A and assign label accordingly
    for i, b in enumerate(predicted_data):
        if any(np.all(a == b) for a in true_data):
            label[i] = 0
    return label

# ----- Seperate Noisy Data -----
def separate_noisy (noisy_data, data_label):
    inlier_data = []
    outlier_data = []

    for data, label in zip(noisy_data, data_label):
        if label == 0:
            inlier_data.append(data)
        elif label == 1:
            outlier_data.append(data)

    return inlier_data, outlier_data

# ----- Paramater Calculation -----
def calculate_metric(predicted_label, true_label, noisy_data):
    
    TP, FP, TN, FN = 0, 0, 0, 0
    TP_data = []
    FP_data = []
    TN_data = []
    FN_data = []
    
    for a, b, data in zip(predicted_label, true_label, noisy_data):
        if np.all(a == 1) and np.all(b == 1):
            TP += 1
            TP_data.append(data)
        elif np.all(a == 1) and np.all(b == 0):
            FP += 1
            FP_data.append(data)
        elif np.all(a == 0) and np.all(b == 1):
            FN += 1
            FN_data.append(data)
        elif np.all(a == 0) and np.all(b == 0):
            TN += 1
            TN_data.append(data)

    if TP + FP + TN + FN != len(true_label):
        raise ValueError ("Metric calculation does not match")

    return TP, FP, TN, FN, TP_data, FP_data, TN_data, FN_data

# ----- Hyperparameter Calculation -----
def MAE(array_1, array_2):

    differences = np.abs(array_1 - array_2)  # Compute absolute differences of y-coordinates
    mae = np.mean(differences)  # Calculate the mean absolute error
    
    return mae

def calculate_hyperparameter(true_label, outliers_label, noisy_data):
    
    TP, FP, TN, FN, TP_data, FP_data, TN_data, FN_data = calculate_metric(outliers_label, true_label, noisy_data)
    
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1_score = 2 / (1 / precision + 1 / recall) if precision != 0 and recall != 0 else 0

    return precision, accuracy, recall, f1_score

# ----- Outlier Detection -----
def calculate_OD_model(true_data, noisy_data, outlier_models, parameter):

    # Step 1: Detect outliers
    if outlier_models == "KDE":
        contamination = parameter[0]
        width = parameter[1]
        outliers, labels = OD_KDE(true_data, noisy_data, contamination, width)
    elif outlier_models == "GMM":
        component = parameter[0]
        tolerance = parameter[1]
        outliers, labels = OD_GMM(true_data, noisy_data, component, tolerance)
    elif outlier_models == "HBOS":
        contamination = parameter[0]
        bin = parameter[1]
        outliers, labels = OD_HBOS(true_data, noisy_data, contamination, bin)
    elif outlier_models == "COPOD":
        contamination = parameter
        outliers, labels = OD_COPOD(true_data, noisy_data, contamination)
    elif outlier_models == "ECOD":
        contamination = parameter
        outliers, labels = OD_ECOD(true_data, noisy_data, contamination)
    
    elif outlier_models == "LMDD":
        contamination = parameter
        outliers, labels = OD_LMDD(true_data, noisy_data, contamination)
    elif outlier_models == "MCD":
        contamination = parameter
        outliers, labels = OD_MCD(true_data, noisy_data, contamination)
    elif outlier_models == "QMCD":
        contamination = parameter
        outliers, labels = OD_QMCD(true_data, noisy_data, contamination)
    elif outlier_models == "PCA":
        contamination = parameter[0]
        component = parameter[1]
        outliers, labels = OD_PCA(true_data, noisy_data, contamination, component)
    elif outlier_models == "KPCA":
        contamination = parameter[0]
        component = parameter[1]
        outliers, labels = OD_KPCA(true_data, noisy_data, contamination, component)
    
    elif outlier_models == "LOF":
        contamination = parameter[0]
        neighbor = parameter[1]
        outliers, labels = OD_LOF(true_data, noisy_data, contamination, neighbor)
    elif outlier_models == "COF":
        contamination = parameter[0]
        neighbor = parameter[1]
        outliers, labels = OD_COF(true_data, noisy_data, contamination, neighbor)
    elif outlier_models == "KNN":
        contamination = parameter[0]
        neighbor = parameter[1]
        outliers, labels = OD_KNN(true_data, noisy_data, contamination, neighbor)
    elif outlier_models == "ABOD":
        contamination = parameter[0]
        neighbor = parameter[1]
        outliers, labels = OD_ABOD(true_data, noisy_data, contamination, neighbor)
    elif outlier_models == "SOS":
        contamination = parameter[0]
        perplexity = parameter[1]
        outliers, labels = OD_SOS(true_data, noisy_data, contamination, perplexity)
    
    elif outlier_models == "FB":
        estimator = parameter[0]
        contamination = parameter[1]
        outliers, labels = OD_FB(true_data, noisy_data, estimator, contamination)
    elif outlier_models == "IF":
        contamination = parameter
        outliers, labels = OD_IF(true_data, noisy_data, contamination)
    elif outlier_models == "LODA":
        contamination = parameter[0]
        bin = parameter[1]
        cut = parameter[2]
        outliers, labels = OD_LODA(true_data, noisy_data, contamination, bin, cut)
    elif outlier_models == "INNE":
        contamination = parameter
        outliers, labels = OD_INNE(true_data, noisy_data, contamination)
    elif outlier_models == "DIF":
        contamination = parameter
        outliers, labels = OD_DIF(true_data, noisy_data, contamination)

    return outliers, labels

def export_results (folder_path, file_name, headers, results):
    file_path = os.path.join(folder_path, file_name)

    # Check if the file exists
    file_exists = os.path.isfile(file_path)

    with open(file_path, 'a') as file:
        writer = csv.DictWriter(file, fieldnames = headers.keys())

        # Write headers if the file doesn't exist
        if not file_exists:
            writer.writeheader()

        writer.writerow(results)

# ----- Load Train-Test IDs from txt file -----
def read_train_test_ids(split_file_path):

    train_file_ids = []
    test_file_ids = []

    with open(split_file_path, "r", encoding = "utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            if "\t" in line:
                value = line.split("\t")[-1].strip()
            elif ":" in line:
                value = line.split(":")[-1].strip()
            else:
                continue

            if line.startswith("Train File IDs"):
                train_file_ids = ast.literal_eval(value)

            elif line.startswith("Test File IDs"):
                test_file_ids = ast.literal_eval(value)

    if len(train_file_ids) == 0 or len(test_file_ids) == 0:
        raise ValueError(f"Could not read train/test IDs from {split_file_path}")

    return train_file_ids, test_file_ids

# ----- Main Execution -----
def main():
    # Production data file path
    downhole_file_path = r"C:\Users\felix\OneDrive\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Outlier Detection Paper\Outlier Detection Simulator\Synthetic Model\downhole_gas_rate.txt"
    
    # Define time and rate raw data
    time_rate, downhole_rate = read_production(downhole_file_path) # hrs., Mcf/d
    time_prod = time_rate[1:]    # hrs.
    downhole_rate = np.array(downhole_rate)
    time_prod = np.array(time_prod) / 24  # days
    log_time_prod = np.log(time_prod)

    # Generate true RNP coordinates
    RNP_data = [Pseudopressure_Conversion.delta_pseudopressure[i] / downhole_rate[i] for i in range(1, len(time_rate))]
    RNP_data = np.array(RNP_data)
    log_RNP_data = np.log(RNP_data)
    log_RNP_coordinates = np.column_stack((log_time_prod, log_RNP_data))

    num_noisy_generate = 10

    # Initialize input values
    true_data = log_RNP_coordinates
    outlier_models = [
        "KDE", "GMM", "HBOS", "COPOD", "ECOD",
        "LMDD", "MCD", "QMCD", "PCA", "KPCA",
        "LOF", "COF", "KNN", "ABOD", "SOS",
        "FB", "IF", "LODA", "INNE", "DIF"
        ]

    # Define outlier model parameters
    FB_estimator = [3]
    FB_contamination = [0.1]
    IF_contamination = [0.3]
    LODA_contamination = [0.05]
    LODA_bin = [30]
    LODA_cut = [150]
    INNE_contamination = [0.3]
    DIF_contamination = [0.15]
    LMDD_contamination = [0.3]
    MCD_contamination = [0.3]
    QMCD_contamination = [0.05]
    PCA_contamination = [0.05]
    PCA_component = [2]
    KPCA_contamination = [0.3]
    KPCA_component = [1]
    KDE_contamination = [0.3]
    KDE_width = [0.1]
    GMM_component = [5]
    GMM_tol = [0.01]
    HBOS_contamination = [0.05]
    HBOS_bin = [30]
    COPOD_contamination = [0.05]
    ECOD_contamination = [0.1]
    LOF_contamination = [0.1]
    LOF_neighbor = [5]
    COF_contamiantion = [0.25]
    COF_neighbor = [20]
    KNN_contamination = [0.15]
    KNN_neighbor = [3]
    ABOD_contamination = [0.05]
    ABOD_neighbor = [5]
    SOS_contamination = [0.3]
    SOS_perplexity = [10]
    
    # Define table headers
    headers = {
        "Data Split": "Data Split",
        "File ID": "File ID",
        "Model": "Model",
        "Input Parameter": "Input Parameter",
        "Parameter Value": "Parameter Value",
        "TP": "TP",
        "TN": "TN",
        "FP": "FP",
        "FN": "FN",
        "MAE": "MAE",
        "Precision": "Precision",
        "Accuracy": "Accuracy",
        "Recall": "Recall",
        "F1 Score": "F1 Score"
    }

    # Define flow regimes & noise level combination
    regimes = ["Transient", "Transition", "BDF", "Transient-Transition", "Transition-BDF", "Transient-BDF", "All"]
    noise_contamination = [25, 50, 75]

    for regime in regimes:
        print("----- Regime:", regime, "-----")
        for contaminate in noise_contamination:
            print("--- Noise level:", contaminate, "---")
            # Specify the folder path and file name for result file
            folder_path = f"C:/Users/felix/OneDrive/Documents/Kuliah/2. Master's Texas A&M University/Publications/Conference Paper/Outlier Detection Paper/Outlier Detection Simulator/OD Results/{regime}/"

            # Create the directory if it doesn't exist
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # Read split file inside each scenario folder
            scenario_folder = f"C:/Users/felix/OneDrive/Documents/Kuliah/2. Master's Texas A&M University/Publications/Conference Paper/Outlier Detection Paper/Outlier Detection Simulator/Noisy RNP Model/{regime}/{contaminate}%"
            split_folder = f"C:/Users/felix/OneDrive/Documents/Kuliah/2. Master's Texas A&M University/Publications/Conference Paper/Smoothing Paper/Smoothing Simulator/Smoothing Methods/Train Results/{regime}/{contaminate}%"
            split_file_path = os.path.join(split_folder, f"train_test_split_{regime}_{contaminate}%.txt")

            train_file_ids, test_file_ids = read_train_test_ids(split_file_path)

            for split_name, file_ids in [("Test", test_file_ids)]:

                print(f"-- {split_name} Files: {file_ids} --")

                file_name = f"OD_Results_{regime}_{contaminate}_{split_name}.txt"
                result_file_path = os.path.join(folder_path, file_name)
                if os.path.exists(result_file_path):
                    os.remove(result_file_path)

                for i in file_ids:
                    print(f"- {split_name} Noisy Data: {i} -")

                    # Read the noisy data
                    noise_file_path = os.path.join(scenario_folder, f"noisy_data_{i}.txt")
                    noisy_time, noisy_RNP = read_noisy(noise_file_path)
                    noisy_time = np.array(noisy_time)
                    noisy_RNP = np.array(noisy_RNP)
                    log_noisy_time = np.log(noisy_time)
                    log_noisy_RNP = np.log(noisy_RNP)
                    noisy_data = np.column_stack((log_noisy_time, log_noisy_RNP))
                    raw_noisy_data = np.column_stack((noisy_time, noisy_RNP))
                    noisy_data_label = data_label(true_data, noisy_data)
                    inlier_data, outlier_data = separate_noisy(raw_noisy_data, noisy_data_label)

                    # # Plot noisy RNP vs time
                    # if regime == "All" and contaminate == 50 and i == 7:
                    #     plt.figure()
                    #     time_inlier = [point[0] for point in inlier_data]
                    #     RNP_inlier = [point[1] for point in inlier_data]
                    #     time_outlier = [point[0] for point in outlier_data]
                    #     RNP_outlier = [point[1] for point in outlier_data]
                    #     plt.plot(time_inlier, RNP_inlier, 'o', label = "inlier")
                    #     plt.plot(time_outlier, RNP_outlier, '^', label = "outlier")
                    #     plt.xlabel("t, days", fontsize = 20)
                    #     plt.ylabel("$\Delta$m(p)/$q_{g}$, $psia^{2}$/cp$\cdot$d/Mscf", fontsize = 20)
                    #     plt.xscale("log")
                    #     plt.yscale("log")
                    #     #plt.title("Rate-normalized Pseudo-pressure (RNP) vs. Time", fontsize = 24)
                    #     plt.minorticks_on()
                    #     plt.tick_params(axis = 'both', which = 'major', labelsize = 18)
                    #     plt.grid(True, which = 'major', linestyle = '-', linewidth = 1)
                    #     plt.grid(True, which = 'minor', linestyle = ':', linewidth = 0.5)
                    #     plt.legend(fontsize = 20)
                    #     plt.show()

                    #     svg_path = r"C:\Users\felix\OneDrive\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Outlier Detection Paper\Outlier Detection Simulator\Figures\Fig_10.svg"
                    #     emf_path = r"C:\Users\felix\OneDrive\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Outlier Detection Paper\Outlier Detection Simulator\Figures\Fig_10.emf"
                    #     inkscape_path = r"C:\Program Files\Inkscape\bin\inkscape.exe"

                    #     convert_svg_to_emf(svg_path, emf_path, inkscape_path = inkscape_path)
        
                    for model in outlier_models:
                        # Record start time
                        start_time = time.time()

                        # Initialize variables for storing the best results
                        best_results = {
                            "Data Split": split_name,
                            "File ID": i,
                            "Model": model,
                            "Input Parameter": None,
                            "Parameter Value": None,
                            "TP": 0,
                            "TN": 0,
                            "FP": 0,
                            "FN": 0,
                            "MAE": float('inf'),
                            "Precision": 0,
                            "Accuracy": 0,
                            "Recall": 0,
                            "F1 Score": 0,
                        }

                        param_values = None
                        input_param = None

                        # Assign parameter values and input parameter
                        if model == "KDE":
                            input_param = "contamination, width"
                            param_values = (KDE_contamination, KDE_width)
                        elif model == "GMM":
                            input_param = "component, tolerance"
                            param_values = (GMM_component, GMM_tol)
                        elif model == "HBOS":
                            input_param = "contamination, bin"
                            param_values = (HBOS_contamination, HBOS_bin)
                        elif model == "COPOD":
                            input_param = "contamination"
                            param_values = COPOD_contamination
                        elif model == "ECOD":
                            input_param = "contamination"
                            param_values = ECOD_contamination
                        
                        elif model == "LMDD":
                            input_param = "contamination"
                            param_values = LMDD_contamination
                        elif model == "MCD":
                            input_param = "contamination"
                            param_values = MCD_contamination
                        elif model == "QMCD":
                            input_param = "contamination"
                            param_values = QMCD_contamination
                        elif model == "PCA":
                            input_param = "contamination, component"
                            param_values = (PCA_contamination, PCA_component)
                        elif model == "KPCA":
                            input_param = "contamination, component"
                            param_values = (KPCA_contamination, KPCA_component)
                        
                        elif model == "LOF":
                            input_param = "contamination, neighbor"
                            param_values = (LOF_contamination, LOF_neighbor)
                        elif model == "COF":
                            input_param = "contamination, neighbor"
                            param_values = (COF_contamiantion, COF_neighbor)
                        elif model == "KNN":
                            input_param = "contamination, neighbor"
                            param_values = (KNN_contamination, KNN_neighbor)
                        elif model == "ABOD":
                            input_param = "contamination, neighbor"
                            param_values = (ABOD_contamination, ABOD_neighbor)
                        elif model == "SOS":
                            input_param = "contamination, perplexity"
                            param_values = (SOS_contamination, SOS_perplexity)
                        
                        elif model == "FB":
                            input_param = "estimator, contamination"
                            param_values = (FB_estimator, FB_contamination)
                        elif model == "IF":
                            input_param = "contamination"
                            param_values = IF_contamination
                        elif model == "LODA":
                            input_param = "contamination, bin, cut"
                            param_values = (LODA_contamination, LODA_bin, LODA_cut)
                        elif model == "INNE":
                            input_param = "contamination"
                            param_values = INNE_contamination
                        elif model == "DIF":
                            input_param = "contamination"
                            param_values = DIF_contamination

                        if param_values is not None:
                            if isinstance(param_values, tuple) and len(param_values) == 2:
                                combine = []
                                for a in param_values[0]:
                                    for b in param_values[1]:
                                        combine.append([a, b])
                                param_values = combine
                            elif isinstance(param_values, tuple) and len(param_values) == 3:
                                combine = []
                                for a in param_values[0]:
                                    for b in param_values[1]:
                                        for c in param_values[2]:
                                            combine.append([a, b, c])
                                param_values = combine
                            else:
                                param_values = param_values

                            for param_value in param_values:

                                outliers, labels = calculate_OD_model(np.array(true_data), np.array(noisy_data), model, param_value)

                                TP, FP, TN, FN, TP_data, FP_data, TN_data, FN_data = calculate_metric(labels, noisy_data_label, raw_noisy_data)

                                mae = mean_absolute_error(noisy_data_label, labels)
                                precision = precision_score(noisy_data_label, labels, zero_division = 0)
                                accuracy = accuracy_score(noisy_data_label, labels)
                                recall = recall_score(noisy_data_label, labels, zero_division = 0)
                                f1 = f1_score(noisy_data_label, labels, zero_division = 0)

                                if f1 > best_results["F1 Score"]:
                                    best_results["Model"] = model
                                    best_results["Input Parameter"] = input_param
                                    best_results["Parameter Value"] = param_value
                                    best_results["TP"] = TP
                                    best_results["TN"] = TN
                                    best_results["FP"] = FP
                                    best_results["FN"] = FN
                                    best_results["MAE"] = mae
                                    best_results["Precision"] = precision
                                    best_results["Accuracy"] = accuracy
                                    best_results["Recall"] = recall
                                    best_results["F1 Score"] = f1

                                # if regime == "All" and contaminate == 50 and i == 7:
                                #     # Plot TP, TN, FP, FN vs time
                                #     plt.figure()
                                #     plt.plot(time_TP, RNP_TP, 'o', label = "TP")
                                #     plt.plot(time_FN, RNP_FN, '^', label = "FN")
                                #     plt.plot(time_TN, RNP_TN, 's', label = "TN")
                                #     plt.plot(time_FP, RNP_FP, 'D', label = "FP")
                                #     plt.xlabel("t, days", fontsize = 20)
                                #     plt.ylabel("$\Delta$m(p)/$q_{g}$, $psia^{2}$/cp$\cdot$d/Mscf", fontsize = 20)
                                #     plt.xscale("log")
                                #     plt.yscale("log")
                                #     #plt.title("Rate-normalized Pseudo-pressure (RNP) vs. Time", fontsize = 24)
                                #     plt.minorticks_on()
                                #     plt.tick_params(axis = 'both', which = 'major', labelsize = 18)
                                #     plt.grid(True, which = 'major', linestyle = '-', linewidth = 1)
                                #     plt.grid(True, which = 'minor', linestyle = ':', linewidth = 0.5)
                                #     plt.legend(fontsize = 20)
                                #     plt.show()

                                #     svg_path = r"C:\Users\felix\OneDrive\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Outlier Detection Paper\Outlier Detection Simulator\Figures\Fig_12.svg"
                                #     emf_path = r"C:\Users\felix\OneDrive\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Outlier Detection Paper\Outlier Detection Simulator\Figures\Fig_12.emf"
                                #     inkscape_path = r"C:\Program Files\Inkscape\bin\inkscape.exe"

                                #     convert_svg_to_emf(svg_path, emf_path, inkscape_path=inkscape_path)

                                #     # Plot TN, FN vs time
                                #     plt.figure()
                                #     time_combined = time_TN.copy()
                                #     time_combined.extend(time_FN)
                                #     RNP_combined = RNP_TN.copy()
                                #     RNP_combined.extend(RNP_FN)
                                #     plt.plot(time_combined, RNP_combined, 'o')
                                #     plt.xlabel("t, days", fontsize = 20)
                                #     plt.ylabel("$\Delta$m(p)/$q_{g}$, $psia^{2}$/cp$\cdot$d/Mscf", fontsize = 20)
                                #     plt.xscale("log")
                                #     plt.yscale("log")
                                #     #plt.title("Rate-normalized Pseudo-pressure (RNP) vs. Time", fontsize = 24)
                                #     plt.minorticks_on()
                                #     plt.tick_params(axis = 'both', which = 'major', labelsize = 18)
                                #     plt.grid(True, which = 'major', linestyle = '-', linewidth = 1)
                                #     plt.grid(True, which = 'minor', linestyle = ':', linewidth = 0.5)
                                #     #plt.legend(fontsize = 20)
                                #     plt.show()

                                #     svg_path = r"C:\Users\felix\OneDrive\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Outlier Detection Paper\Outlier Detection Simulator\Figures\Fig_13.svg"
                                #     emf_path = r"C:\Users\felix\OneDrive\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Outlier Detection Paper\Outlier Detection Simulator\Figures\Fig_13.emf"
                                #     inkscape_path = r"C:\Program Files\Inkscape\bin\inkscape.exe"

                                #     convert_svg_to_emf(svg_path, emf_path, inkscape_path=inkscape_path)
        
                        export_results(folder_path, file_name, headers, best_results)
                        duration = time.time() - start_time
                        print("-> Model Proceed:", model, "(",duration,"secs.)")

    print("----- Outlier Detection Simulator Success -----")
    return

main()