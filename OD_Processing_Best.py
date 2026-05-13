# -*- coding: utf-8 -*-
"""
Outlier Detection Models

Author: Munthe, Felix A.
Created on Friday, 23 June 2023
"""

import os
import csv
import time
import Pseudopressure_Conversion
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, accuracy_score, recall_score, precision_score, f1_score
from OD_Models_Best import (
    OD_MCD, OD_LOF, OD_FB, OD_GMM
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
def calculate_OD_model(true_data, noisy_data, outlier_models, parameter, return_scores = False):

    # Step 1: Detect outliers
    if outlier_models == "GMM":
        component = parameter[0]
        tolerance = parameter[1]
        if return_scores:
            outliers, labels, scores, threshold = OD_GMM(true_data, noisy_data, component, tolerance, return_scores = True)
            return outliers, labels, scores, threshold
        outliers, labels = OD_GMM(true_data, noisy_data, component, tolerance)

    elif outlier_models == "MCD":
        contamination = parameter
        if return_scores:
            outliers, labels, scores, threshold = OD_MCD(true_data, noisy_data, contamination, return_scores = True)
            return outliers, labels, scores, threshold
        outliers, labels = OD_MCD(true_data, noisy_data, contamination)

    elif outlier_models == "LOF":
        contamination = parameter[0]
        neighbor = parameter[1]
        if return_scores:
            outliers, labels, scores, threshold = OD_LOF(true_data, noisy_data, contamination, neighbor, return_scores = True)
            return outliers, labels, scores, threshold
        outliers, labels = OD_LOF(true_data, noisy_data, contamination, neighbor)

    elif outlier_models == "FB":
        estimator = parameter[0]
        contamination = parameter[1]
        if return_scores:
            outliers, labels, scores, threshold = OD_FB(true_data, noisy_data, estimator, contamination, return_scores = True)
            return outliers, labels, scores, threshold
        outliers, labels = OD_FB(true_data, noisy_data, estimator, contamination)

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

def export_confusion_points(folder_path, file_name, TP_data, FP_data, TN_data, FN_data):
    file_path = os.path.join(folder_path, file_name)

    with open(file_path, "w", newline = "") as file:
        writer = csv.writer(file)
        writer.writerow(["Class", "Time_days", "RNP"])

        for t, r in TP_data:
            writer.writerow(["TP", t, r])
        for t, r in FP_data:
            writer.writerow(["FP", t, r])
        for t, r in TN_data:
            writer.writerow(["TN", t, r])
        for t, r in FN_data:
            writer.writerow(["FN", t, r])

def export_case_metrics(folder_path, file_name, case_outputs):
    file_path = os.path.join(folder_path, file_name)

    with open(file_path, "w", newline = "") as file:
        writer = csv.writer(file)
        writer.writerow(["Model", "Parameter Value", "TP", "TN", "FP", "FN"])

        for model_name, d in case_outputs.items():
            writer.writerow([
                model_name,
                d["params"],
                d["TP"],
                d["TN"],
                d["FP"],
                d["FN"],
            ])

def plot_score_comparison_figure(noisy_time, case_outputs, svg_path, emf_path, inkscape_path):
    fig, axes = plt.subplots(2, 2, figsize = (14, 10), sharex = True)
    model_order = ["MCD", "LOF", "FB", "GMM"]

    for ax, model_name in zip(axes.flatten(), model_order):
        scores = np.array(case_outputs[model_name]["scores"])
        labels = np.array(case_outputs[model_name]["labels"])
        threshold = case_outputs[model_name]["threshold"]

        ax.plot(noisy_time, scores, 'o', markersize = 4, label = "Score")
        ax.plot(noisy_time[labels == 1], scores[labels == 1], '^', markersize = 6, label = "Predicted outlier")

        if threshold is not None:
            ax.axhline(threshold, linestyle = '--', linewidth = 1, label = "Threshold")

        ax.set_xscale("log")
        ax.set_title(model_name, fontsize = 18)
        ax.minorticks_on()
        ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
        ax.grid(True, which = 'major', linestyle = '-', linewidth = 1)
        ax.grid(True, which = 'minor', linestyle = ':', linewidth = 0.5)
        

    axes[1, 0].set_xlabel("$t$, day", fontsize = 18)
    axes[1, 1].set_xlabel("$t$, day", fontsize = 18)
    axes[0, 0].set_ylabel("Outlier score", fontsize = 18)
    axes[1, 0].set_ylabel("Outlier score", fontsize = 18)

    handles, labels_legend = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc = "upper center", ncol = 3, fontsize = 14)
    fig.tight_layout(rect = [0, 0, 1, 0.94])

    plt.show()
    convert_svg_to_emf(svg_path, emf_path, inkscape_path = inkscape_path)

def fit_fixed_slope_intercept(time_vals, rnp_vals, slope = 0.5):
    time_vals = np.asarray(time_vals, dtype = float)
    rnp_vals = np.asarray(rnp_vals, dtype = float)

    log_t = np.log10(time_vals)
    log_rnp = np.log10(rnp_vals)

    intercept = np.mean(log_rnp - slope * log_t)
    return intercept

def evaluate_fixed_slope_window(time_vals, rnp_vals, slope = 0.5):
    intercept = fit_fixed_slope_intercept(time_vals, rnp_vals, slope = slope)

    log_t = np.log10(time_vals)
    log_rnp = np.log10(rnp_vals)
    log_pred = intercept + slope * log_t

    residuals = log_rnp - log_pred
    rmse = np.sqrt(np.mean(residuals ** 2))

    return intercept, rmse

def search_best_fixed_slope_interval(
    time_vals,
    rnp_vals,
    slope=0.5,
    min_points=8,
    min_log_span=0.8,
    t_min=None,
    t_max=None,
    rmse_tolerance=0.10
    ):
    """
    Search for the best contiguous interval for a fixed slope in log-log space,
    while favoring longer intervals.

    Selection logic:
    1. Compute RMSE for all valid intervals
    2. Keep intervals within rmse_tolerance of the minimum RMSE
    3. Among those, choose the one with the largest log-span
    4. If still tied, choose the one with the most points
    5. If still tied, choose the one with the lowest RMSE
    """

    time_vals = np.asarray(time_vals, dtype=float)
    rnp_vals = np.asarray(rnp_vals, dtype=float)

    mask = np.isfinite(time_vals) & np.isfinite(rnp_vals) & (time_vals > 0) & (rnp_vals > 0)

    if t_min is not None:
        mask &= time_vals >= t_min
    if t_max is not None:
        mask &= time_vals <= t_max

    time_vals = time_vals[mask]
    rnp_vals = rnp_vals[mask]

    sort_idx = np.argsort(time_vals)
    time_vals = time_vals[sort_idx]
    rnp_vals = rnp_vals[sort_idx]

    n = len(time_vals)
    candidates = []

    for i in range(n):
        for j in range(i + min_points - 1, n):
            t_window = time_vals[i:j+1]
            rnp_window = rnp_vals[i:j+1]

            log_span = np.log10(t_window[-1]) - np.log10(t_window[0])
            if log_span < min_log_span:
                continue

            intercept, rmse = evaluate_fixed_slope_window(
                t_window,
                rnp_window,
                slope=slope
            )

            candidates.append({
                "start_idx": i,
                "end_idx": j,
                "start_time": t_window[0],
                "end_time": t_window[-1],
                "n_points": len(t_window),
                "log_span": log_span,
                "intercept": intercept,
                "rmse": rmse,
            })

    if not candidates:
        raise ValueError("No valid interval found. Try reducing min_points or min_log_span.")

    # Step 1: best RMSE
    min_rmse = min(c["rmse"] for c in candidates)

    # Step 2: keep intervals close to the best RMSE
    rmse_cutoff = min_rmse * (1 + rmse_tolerance)
    filtered = [c for c in candidates if c["rmse"] <= rmse_cutoff]

    # Step 3: among acceptable fits, prefer longer intervals
    # Step 4: then prefer more points
    # Step 5: then lower RMSE
    best = sorted(
        filtered,
        key=lambda c: (-c["log_span"], -c["n_points"], c["rmse"])
    )[0]

    # Build fitted line
    t_line = np.logspace(np.log10(best["start_time"]), np.log10(best["end_time"]), 200)
    rnp_line = 10 ** (best["intercept"] + slope * np.log10(t_line))

    best["line_t"] = t_line
    best["line_rnp"] = rnp_line
    best["slope"] = slope
    best["min_rmse"] = min_rmse
    best["rmse_cutoff"] = rmse_cutoff
    best["n_candidates"] = len(candidates)
    best["n_filtered"] = len(filtered)

    return best

def plot_fig14_with_auto_intervals(
    noisy_time,
    noisy_rnp,
    best_noisy,
    best_clean,
    svg_path = None,
    emf_path = None,
    inkscape_path = None
    ):
    plt.figure(figsize = (14, 8))
    plt.plot(noisy_time, noisy_rnp, 'o', markersize = 4, alpha = 0.9)

    # solid red line = best interval from raw noisy data
    plt.plot(best_noisy["line_t"], best_noisy["line_rnp"], 'r-', linewidth = 2.5)

    # dashed red line = best interval from cleaned data
    plt.plot(best_clean["line_t"], best_clean["line_rnp"], 'r--', linewidth = 2.5)

    plt.xlabel("$t$, day", fontsize = 20)
    plt.ylabel("$\\Delta$m(p)/$q_{g}$, $psia^{2}$/cp$\\cdot$d/Mscf", fontsize = 20)
    plt.xscale("log")
    plt.yscale("log")
    plt.minorticks_on()
    plt.tick_params(axis = 'both', which = 'major', labelsize = 18)
    plt.grid(True, which = 'major', linestyle = '-', linewidth = 1)
    plt.grid(True, which = 'minor', linestyle = ':', linewidth = 0.5)

    plt.show()

    if svg_path is not None and emf_path is not None and inkscape_path is not None:
        convert_svg_to_emf(svg_path, emf_path, inkscape_path = inkscape_path)

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
        "MCD", "LOF", "FB", "GMM"
        ]

    # Define outlier model parameters
    component_GMM = [5]
    tolerance_GMM = [0.01]
    estimator_FB = [3]
    contamination_FB = [0.1]
    contamination_LOF = [0.1]
    neighbor_LOF = [5]
    contamination_MCD = [0.3]

    
    # Define table headers
    headers = {
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
    regimes = ["All"] # ["Transient", "Transition", "BDF", "Transient-Transition", "Transition-BDF", "Transient-BDF", "All"]
    noise_contamination = [50] # [25, 50, 75]

    for regime in regimes:
        print("----- Regime:", regime, "-----")
        for contaminate in noise_contamination:
            print("--- Noise level:", contaminate, "---")
            # Specify the folder path and file name for result file
            folder_path = f"C:/Users/felix/OneDrive/Documents/Kuliah/2. Master's Texas A&M University/Publications/Conference Paper/Outlier Detection Paper/Outlier Detection Simulator/OD Results/Top Four/{regime}/"
            file_name = f"OD_Results_{regime}_{contaminate}.txt"

            # Create the directory if it doesn't exist
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # Process the OD over the range of noisy data
            for i in range (9, 10): # 3, 6, 9
                print("- Noisy Data:", i, "-")
                
                # Read the noisy data
                noise_file_path = f"C:/Users/felix/OneDrive/Documents/Kuliah/2. Master's Texas A&M University/Publications/Conference Paper/Outlier Detection Paper/Outlier Detection Simulator/Noisy RNP Model/{regime}/{contaminate}%/noisy_data_{i}.txt"
                noisy_time, noisy_RNP = read_noisy(noise_file_path)
                noisy_time = np.array(noisy_time)
                noisy_RNP = np.array(noisy_RNP)
                log_noisy_time = np.log(noisy_time)
                log_noisy_RNP = np.log(noisy_RNP)
                noisy_data = np.column_stack((log_noisy_time, log_noisy_RNP))
                raw_noisy_data = np.column_stack((noisy_time, noisy_RNP))
                noisy_data_label = data_label(true_data, noisy_data)
                inlier_data, outlier_data = separate_noisy(raw_noisy_data, noisy_data_label)

                best_case_outputs = {}

                # Plot noisy RNP vs time
                if regime == "All" and contaminate == 50:
                    plt.figure()
                    time_inlier = [point[0] for point in inlier_data]
                    RNP_inlier = [point[1] for point in inlier_data]
                    time_outlier = [point[0] for point in outlier_data]
                    RNP_outlier = [point[1] for point in outlier_data]
                    plt.plot(time_inlier, RNP_inlier, 'o', label = "inlier")
                    plt.plot(time_outlier, RNP_outlier, '^', label = "outlier")
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

                    svg_path = r"C:\Users\felix\OneDrive\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Smoothing Paper\Smoothing Simulator\Figures\Fig_11.svg"
                    emf_path = r"C:\Users\felix\OneDrive\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Smoothing Paper\Smoothing Simulator\Figures\Fig_11.emf"
                    inkscape_path = r"C:\Program Files\Inkscape\bin\inkscape.exe"

                    # svg_path = r"C:\Users\felix\OneDrive\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Outlier Detection Paper\Outlier Detection Simulator\Figures\Fig_10.svg"
                    # emf_path = r"C:\Users\felix\OneDrive\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Outlier Detection Paper\Outlier Detection Simulator\Figures\Fig_10.emf"
                    # inkscape_path = r"C:\Program Files\Inkscape\bin\inkscape.exe"

                    convert_svg_to_emf(svg_path, emf_path, inkscape_path = inkscape_path)
        
                for model in outlier_models:
                    # Record start time
                    start_time = time.time()

                    # Initialize variables for storing the best results
                    best_results = {
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
                    if model == "GMM":
                        input_param = "component, tolerance"
                        param_values = (component_GMM, tolerance_GMM)
                    
                    elif model == "MCD":
                        input_param = "contamination"
                        param_values = contamination_MCD
                    
                    elif model == "LOF":
                        input_param = "contamination, neighbor"
                        param_values = (contamination_LOF, neighbor_LOF)
                    
                    elif model == "FB":
                        input_param = "estimator, contamination"
                        param_values = (estimator_FB, contamination_FB)
            
                    if param_values is not None:
                        if isinstance (param_values, tuple) and len(param_values) == 2:
                            combine = []
                            for a in param_values[0]:
                                for b in param_values[1]:
                                    combine.append([a, b])
                            param_values = combine
                        elif isinstance (param_values, tuple) and len(param_values) == 3:
                            combine = []
                            for a in param_values[0]:
                                for b in param_values[1]:
                                    for c in param_values[2]:
                                        combine.append([a, b, c])
                            param_values = combine
                        else:
                            param_values = param_values

                        for param_value in param_values:
                            
                            # Perform outlier detection and evaluation
                            outliers, labels, scores, threshold = calculate_OD_model(np.array(true_data), np.array(noisy_data), model, param_value, return_scores = True)
                                
                            # Calculate TP, TN, FP, FN
                            TP, FP, TN, FN, TP_data, FP_data, TN_data, FN_data = calculate_metric(labels, noisy_data_label, raw_noisy_data)

                            time_TP = [point[0] for point in TP_data]
                            RNP_TP = [point[1] for point in TP_data]
                            time_FP = [point[0] for point in FP_data]
                            RNP_FP = [point[1] for point in FP_data]
                            time_TN = [point[0] for point in TN_data]
                            RNP_TN = [point[1] for point in TN_data]
                            time_FN = [point[0] for point in FN_data]
                            RNP_FN = [point[1] for point in FN_data]
                    
                            # Calculate MAE, precision, accuracy, recall, and F1 score
                            mae = mean_absolute_error(noisy_data_label, labels)
                            #precision, accuracy, recall, f1_score = calculate_hyperparameter(noisy_data_label, labels, noisy_data)
                            precision = precision_score(noisy_data_label, labels)
                            accuracy = accuracy_score(noisy_data_label, labels)
                            recall = recall_score(noisy_data_label, labels)
                            f1 = f1_score(noisy_data_label, labels)

                            # Update best results if current F1 score is higher
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

                                if regime == "All" and contaminate == 50:
                                    best_case_outputs[model] = {
                                        "scores": np.array(scores),
                                        "threshold": threshold,
                                        "labels": np.array(labels),
                                        "params": param_value,
                                        "TP": TP,
                                        "TN": TN,
                                        "FP": FP,
                                        "FN": FN,
                                    }
                                    
                                    if model == "LOF":
                                        # Export true RNP data
                                        file_path = "C:/Users/felix/OneDrive/Documents/Kuliah/2. Master's Texas A&M University/Publications/Conference Paper/Outlier Detection Paper/Outlier Detection Simulator/OD Results/TP_results.txt"
                                        with open(file_path, "w") as file:
                                            # Write the header
                                            file.write("t(day)\tRNP(psia2/cp)\n")
                                            # Write the data to the file
                                            for x, y in zip(time_TP, RNP_TP):
                                                file.write(f"{x}\t{y}\n")
                                        file_path = "C:/Users/felix/OneDrive/Documents/Kuliah/2. Master's Texas A&M University/Publications/Conference Paper/Outlier Detection Paper/Outlier Detection Simulator/OD Results/FN_results.txt"
                                        with open(file_path, "w") as file:
                                            # Write the header
                                            file.write("t(day)\tRNP(psia2/cp)\n")
                                            # Write the data to the file
                                            for x, y in zip(time_FN, RNP_FN):
                                                file.write(f"{x}\t{y}\n")
                                        file_path = "C:/Users/felix/OneDrive/Documents/Kuliah/2. Master's Texas A&M University/Publications/Conference Paper/Outlier Detection Paper/Outlier Detection Simulator/OD Results/TN_results.txt"
                                        with open(file_path, "w") as file:
                                            # Write the header
                                            file.write("t(day)\tRNP(psia2/cp)\n")
                                            # Write the data to the file
                                            for x, y in zip(time_TN, RNP_TN):
                                                file.write(f"{x}\t{y}\n")
                                        file_path = "C:/Users/felix/OneDrive/Documents/Kuliah/2. Master's Texas A&M University/Publications/Conference Paper/Outlier Detection Paper/Outlier Detection Simulator/OD Results/FP_results.txt"
                                        with open(file_path, "w") as file:
                                            # Write the header
                                            file.write("t(day)\tRNP(psia2/cp)\n")
                                            # Write the data to the file
                                            for x, y in zip(time_FP, RNP_FP):
                                                file.write(f"{x}\t{y}\n")

                                        # Plot TP, TN, FP, FN vs time
                                        plt.figure()
                                        plt.plot(time_TP, RNP_TP, 'o', label = "TP")
                                        plt.plot(time_FN, RNP_FN, '^', label = "FN")
                                        plt.plot(time_TN, RNP_TN, 's', label = "TN")
                                        plt.plot(time_FP, RNP_FP, 'D', label = "FP")
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

                                        svg_path = r"C:\Users\felix\OneDrive\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Outlier Detection Paper\Outlier Detection Simulator\Figures\Fig_12.svg"
                                        emf_path = r"C:\Users\felix\OneDrive\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Outlier Detection Paper\Outlier Detection Simulator\Figures\Fig_12.emf"
                                        inkscape_path = r"C:\Program Files\Inkscape\bin\inkscape.exe"

                                        # convert_svg_to_emf(svg_path, emf_path, inkscape_path=inkscape_path)

                                        # Plot TN, FN vs time
                                        plt.figure()
                                        time_combined = time_TN.copy()
                                        time_combined.extend(time_FN)
                                        RNP_combined = RNP_TN.copy()
                                        RNP_combined.extend(RNP_FN)
                                        plt.plot(time_combined, RNP_combined, 'o')
                                        plt.xlabel("$t$, day", fontsize = 20)
                                        plt.ylabel("$\Delta$m(p)/$q_{g}$, $psia^{2}$/cp$\cdot$d/Mscf", fontsize = 20)
                                        plt.xscale("log")
                                        plt.yscale("log")
                                        #plt.title("Rate-normalized Pseudo-pressure (RNP) vs. Time", fontsize = 24)
                                        plt.minorticks_on()
                                        plt.tick_params(axis = 'both', which = 'major', labelsize = 18)
                                        plt.grid(True, which = 'major', linestyle = '-', linewidth = 1)
                                        plt.grid(True, which = 'minor', linestyle = ':', linewidth = 0.5)
                                        #plt.legend(fontsize = 20)
                                        plt.show()

                                        svg_path = r"C:\Users\felix\OneDrive\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Outlier Detection Paper\Outlier Detection Simulator\Figures\Fig_13.svg"
                                        emf_path = r"C:\Users\felix\OneDrive\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Outlier Detection Paper\Outlier Detection Simulator\Figures\Fig_13.emf"
                                        inkscape_path = r"C:\Program Files\Inkscape\bin\inkscape.exe"

                                        # convert_svg_to_emf(svg_path, emf_path, inkscape_path = inkscape_path)

                                        # Build the LOF-retained dataset (same logic as your Fig. 13)
                                        time_clean = time_TN.copy()
                                        time_clean.extend(time_FN)

                                        RNP_clean = RNP_TN.copy()
                                        RNP_clean.extend(RNP_FN)

                                        best_noisy = search_best_fixed_slope_interval(
                                            noisy_time,
                                            noisy_RNP,
                                            slope = 0.5,
                                            min_points = 10,
                                            min_log_span = 1.0,
                                            t_min = 1,
                                            t_max = 4000,
                                            rmse_tolerance = 0.15
                                        )

                                        best_clean = search_best_fixed_slope_interval(
                                            time_clean,
                                            RNP_clean,
                                            slope = 0.5,
                                            min_points = 10,
                                            min_log_span = 1.0,
                                            t_min = 1,
                                            t_max = 4000,
                                            rmse_tolerance = 0.15
                                        )

                                        print("Best noisy interval:", best_noisy["start_time"], "to", best_noisy["end_time"], "day")
                                        print("Best clean interval:", best_clean["start_time"], "to", best_clean["end_time"], "day")
                                        print("Noisy RMSE:", best_noisy["rmse"], "| cutoff:", best_noisy["rmse_cutoff"])
                                        print("Clean RMSE:", best_clean["rmse"], "| cutoff:", best_clean["rmse_cutoff"])

                                        plot_fig14_with_auto_intervals(
                                            noisy_time=noisy_time,
                                            noisy_rnp=noisy_RNP,
                                            best_noisy=best_noisy,
                                            best_clean=best_clean,
                                            svg_path=svg_path,
                                            emf_path=emf_path,
                                            inkscape_path=inkscape_path
                                        )
                                        svg_path = r"C:\Users\felix\OneDrive\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Outlier Detection Paper\Outlier Detection Simulator\Figures\Fig_14.svg"
                                        emf_path = r"C:\Users\felix\OneDrive\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Outlier Detection Paper\Outlier Detection Simulator\Figures\Fig_14.emf"
                                        inkscape_path = r"C:\Program Files\Inkscape\bin\inkscape.exe"

                                        # convert_svg_to_emf(svg_path, emf_path, inkscape_path = inkscape_path)
        
                #     export_results(folder_path, file_name, headers, best_results)
                #     duration = time.time() - start_time
                #     print("-> Model Proceed:", model, "(",duration,"secs.)")

                # if regime == "All" and contaminate == 50 and len(best_case_outputs) == 4:
                #     svg_path = r"C:\Users\felix\OneDrive\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Outlier Detection Paper\Outlier Detection Simulator\Figures\Fig_11.svg"
                #     emf_path = r"C:\Users\felix\OneDrive\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Outlier Detection Paper\Outlier Detection Simulator\Figures\Fig_11.emf"
                #     inkscape_path = r"C:\Program Files\Inkscape\bin\inkscape.exe"

                #     plot_score_comparison_figure(
                #         noisy_time,
                #         best_case_outputs,
                #         svg_path,
                #         emf_path,
                #         inkscape_path
                #     )

    print("----- Outlier Detection Simulator Success -----")
    return

main()