# -*- coding: utf-8 -*-
"""
Outlier Detection Models

Author: Munthe, Felix A.
Created on Friday, 23 June 2023
"""

from pyod.models.kde import KDE
from pyod.models.gmm import GMM
from pyod.models.hbos import HBOS
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD

from pyod.models.lmdd import LMDD
from pyod.models.mcd import MCD
from pyod.models.qmcd import QMCD
from pyod.models.pca import PCA
from pyod.models.kpca import KPCA

from pyod.models.lof import LOF
from pyod.models.cof import COF
from pyod.models.knn import KNN
from pyod.models.abod import ABOD
from pyod.models.sos import SOS

from pyod.models.feature_bagging import FeatureBagging
from pyod.models.iforest import IForest
from pyod.models.inne import INNE
from pyod.models.loda import LODA
from pyod.models.dif import DIF

# ----- Probabilistic -----
def OD_KDE(true_data, noisy_data, contamination, width):  # Kernel Density Estimation (Latecki et al., 2007)

    # Step 1: Create and train the model with specified parameters
    model = KDE(contamination = contamination, bandwidth = width)
    model.fit(true_data)
    
    # Step 2: Predict outliers
    y_pred = model.predict(noisy_data)
    
    # Step 3: Get the predicted labels
    labels = y_pred

    # Step 4: Separate inliers and outliers for plotting
    inliers = noisy_data[labels == 0]
    outliers = noisy_data[labels == 1]

    return outliers, labels

def OD_GMM(true_data, noisy_data, component, tolerance): # Gaussian Mixture Model (Reynolds, 2009; Aggarwal, 2015)

    # Step 1: Create and train the model with specified parameters
    model = GMM(n_components = component, tol = tolerance)
    model.fit(true_data)
    
    # Step 2: Predict outliers
    y_pred = model.predict(noisy_data)
    
    # Step 3: Get the predicted labels
    labels = y_pred

    # Step 4: Separate inliers and outliers for plotting
    inliers = noisy_data[labels == 0]
    outliers = noisy_data[labels == 1]

    return outliers, labels

def OD_HBOS(true_data, noisy_data, contamination, bin): # Histogram-Based Outlier Score (Goldstein and Dengel, 2012)

    # Step 1: Create and train the model with specified parameters
    model = HBOS(contamination = contamination, n_bins = bin)
    model.fit(true_data)
    
    # Step 2: Predict outliers
    y_pred = model.predict(noisy_data)
    
    # Step 3: Get the predicted labels
    labels = y_pred

    # Step 4: Separate inliers and outliers for plotting
    inliers = noisy_data[labels == 0]
    outliers = noisy_data[labels == 1]

    return outliers, labels

def OD_COPOD(true_data, noisy_data, contamination): # Copula-based Outlier Detection (Li et al., 2020)

    # Step 1: Create and train the model with specified parameters
    model = COPOD(contamination = contamination)
    model.fit(true_data)
    
    # Step 2: Predict outliers
    y_pred = model.predict(noisy_data)
    
    # Step 3: Get the predicted labels
    labels = y_pred

    # Step 4: Separate inliers and outliers for plotting
    inliers = noisy_data[labels == 0]
    outliers = noisy_data[labels == 1]

    return outliers, labels

def OD_ECOD(true_data, noisy_data, contamination): # Unsupervised Outlier Detection Using Empirical Cumulative Distribution Functions (Li et al., 2022)
    
    # Step 1: Create and train the model with specified parameters
    model = ECOD(contamination = contamination)
    model.fit(true_data)
    
    # Step 2: Predict outliers
    y_pred = model.predict(noisy_data)
    
    # Step 3: Get the predicted labels
    labels = y_pred

    # Step 4: Separate inliers and outliers for plotting
    inliers = noisy_data[labels == 0]
    outliers = noisy_data[labels == 1]

    return outliers, labels

# ----- Linear Model -----
def OD_LMDD(true_data, noisy_data, contamination): # Linear Model for Deviation Detection (Arning, Agrawal, and Raghavan, 1996)
    
    # Step 1: Create and train the model with specified parameters
    model = LMDD(contamination = contamination)
    model.fit(true_data)
    
    # Step 2: Predict outliers
    y_pred = model.predict(noisy_data)
    
    # Step 3: Get the predicted labels
    labels = y_pred

    # Step 4: Separate inliers and outliers for plotting
    inliers = noisy_data[labels == 0]
    outliers = noisy_data[labels == 1]

    return outliers, labels

def OD_MCD(true_data, noisy_data, contamination): # Minimum Covariance Determinant (Rousseeuw and Van Driessen, 1999)

    # Step 1: Create and train the model with specified parameters
    model = MCD(contamination = contamination)
    model.fit(true_data)
    
    # Step 2: Predict outliers
    y_pred = model.predict(noisy_data)
    
    # Step 3: Get the predicted labels
    labels = y_pred

    # Step 4: Separate inliers and outliers for plotting
    inliers = noisy_data[labels == 0]
    outliers = noisy_data[labels == 1]

    return outliers, labels

def OD_QMCD(true_data, noisy_data, contamination): # Quasi-Monte Carlo Discrepancy (2001)
    
    # Step 1: Create and train the model with specified parameters
    model = QMCD(contamination = contamination)
    model.fit(true_data)
    
    # Step 2: Predict outliers
    y_pred = model.predict(noisy_data)
    
    # Step 3: Get the predicted labels
    labels = y_pred

    # Step 4: Separate inliers and outliers for plotting
    inliers = noisy_data[labels == 0]
    outliers = noisy_data[labels == 1]

    return outliers, labels

def OD_PCA(true_data, noisy_data, contamination, component): # Principal Component Analysis (Shyu et al., 2003)

    # Step 1: Create and train the model with specified parameters
    model = PCA(contamination = contamination, n_components = component)
    model.fit(true_data)
    
    # Step 2: Predict outliers
    y_pred = model.predict(noisy_data)
    
    # Step 3: Get the predicted labels
    labels = y_pred

    # Step 4: Separate inliers and outliers for plotting
    inliers = noisy_data[labels == 0]
    outliers = noisy_data[labels == 1]

    return outliers, labels

def OD_KPCA(true_data, noisy_data, contamination, component): # Kernel PCA (Hoffmann, 2007)
    
    # Step 1: Create and train the model with specified parameters
    model = KPCA(contamination = contamination, n_components = component)
    model.fit(true_data)
    
    # Step 2: Predict outliers
    y_pred = model.predict(noisy_data)
    
    # Step 3: Get the predicted labels
    labels = y_pred

    # Step 4: Separate inliers and outliers for plotting
    inliers = noisy_data[labels == 0]
    outliers = noisy_data[labels == 1]

    return outliers, labels

# ----- Proximity-based -----
def OD_LOF(true_data, noisy_data, contamination, neighbor): # Local Outlier Factor (Breunig et al., 2000)
    
    # Step 1: Create and train the model with specified parameters
    model = LOF(contamination = contamination, n_neighbors = neighbor)
    model.fit(true_data)
    
    # Step 2: Predict outliers
    y_pred = model.predict(noisy_data)
    
    # Step 3: Get the predicted labels
    labels = y_pred

    # Step 4: Separate inliers and outliers for plotting
    inliers = noisy_data[labels == 0]
    outliers = noisy_data[labels == 1]

    return outliers, labels

def OD_COF(true_data, noisy_data, contamination, neighbor): # Connectivity-Based Outlier Factor (Tang et al., 2002)

    # Step 1: Create and train the model with specified parameters
    model = COF(contamination = contamination, n_neighbors = neighbor)
    model.fit(true_data)
    
    # Step 2: Predict outliers
    y_pred = model.predict(noisy_data)
    
    # Step 3: Get the predicted labels
    labels = y_pred

    # Step 4: Separate inliers and outliers for plotting
    inliers = noisy_data[labels == 0]
    outliers = noisy_data[labels == 1]

    return outliers, labels

def OD_KNN(true_data, noisy_data, contamination, neighbor): # k Nearest Neighbors (Angiulli and Pizzuti, 2002)

    # Step 1: Create and train the model with specified parameters
    model = KNN(contamination = contamination, n_neighbors = neighbor)
    model.fit(true_data)
    
    # Step 2: Predict outliers
    y_pred = model.predict(noisy_data)
    
    # Step 3: Get the predicted labels
    labels = y_pred

    # Step 4: Separate inliers and outliers for plotting
    inliers = noisy_data[labels == 0]
    outliers = noisy_data[labels == 1]

    return outliers, labels

def OD_ABOD(true_data, noisy_data, contamination, neighbor): # Angle-based Outlier Detection (Kriegel, Schubert, and Zimek, 2008)
    
    # Step 1: Create and train the model with specified parameters
    model = ABOD(contamination = contamination, n_neighbors = neighbor)
    model.fit(true_data)
    
    # Step 2: Predict outliers
    y_pred = model.predict(noisy_data)
    
    # Step 3: Get the predicted labels
    labels = y_pred

    # Step 4: Separate inliers and outliers for plotting
    inliers = noisy_data[labels == 0]
    outliers = noisy_data[labels == 1]

    return outliers, labels

def OD_SOS(true_data, noisy_data, contamination, perplexity): # Stochastic Outlier Selection (Janssens et al., 2012)

    # Step 1: Create and train the model with specified parameters
    model = SOS(contamination = contamination, perplexity = perplexity)
    model.fit(true_data)
    
    # Step 2: Predict outliers
    y_pred = model.predict(noisy_data)
    
    # Step 3: Get the predicted labels
    labels = y_pred

    # Step 4: Separate inliers and outliers for plotting
    inliers = noisy_data[labels == 0]
    outliers = noisy_data[labels == 1]

    return outliers, labels

# ----- Ensembles -----
def OD_FB(true_data, noisy_data, estimator, contamination): # Feature Bagging (Lazarevic & Kumar, 2005)
    
    # Step 1: Create and train the model with specified parameters
    model = FeatureBagging(n_estimators = estimator, contamination = contamination)
    model.fit(true_data)
    
    # Step 2: Predict outliers
    y_pred = model.predict(noisy_data)
    
    # Step 3: Get the predicted labels
    labels = y_pred

    # Step 4: Separate inliers and outliers for plotting
    inliers = noisy_data[labels == 0]
    outliers = noisy_data[labels == 1]

    return outliers, labels

def OD_IF(true_data, noisy_data, contamination):  # Isolation Forest (Liu et al., 2008)

    # Step 1: Create and train the model with specified parameters
    model = IForest(contamination = contamination)
    model.fit(true_data)
    
    # Step 2: Predict outliers
    y_pred = model.predict(noisy_data)
    
    # Step 3: Get the predicted labels
    labels = y_pred

    # Step 4: Separate inliers and outliers for plotting
    inliers = noisy_data[labels == 0]
    outliers = noisy_data[labels == 1]

    return outliers, labels

def OD_LODA(true_data, noisy_data, contamination, bin, cut): # Lightweight On-line Detector of Anomalies (Pevný, 2016)

    # Step 1: Create and train the model with specified parameters
    model = LODA(contamination = contamination, n_bins = bin, n_random_cuts = cut)
    model.fit(true_data)
    
    # Step 2: Predict outliers
    y_pred = model.predict(noisy_data)
    
    # Step 3: Get the predicted labels
    labels = y_pred

    # Step 4: Separate inliers and outliers for plotting
    inliers = noisy_data[labels == 0]
    outliers = noisy_data[labels == 1]

    return outliers, labels

def OD_INNE(true_data, noisy_data, contamination): # Isolation-based Anomaly Detection Using Nearest-Neighbor Ensembles (Bandaragoda et al., 2018)

    # Step 1: Create and train the model with specified parameters
    model = INNE(contamination = contamination)
    model.fit(true_data)
    
    # Step 2: Predict outliers
    y_pred = model.predict(noisy_data)
    
    # Step 3: Get the predicted labels
    labels = y_pred

    # Step 4: Separate inliers and outliers for plotting
    inliers = noisy_data[labels == 0]
    outliers = noisy_data[labels == 1]

    return outliers, labels

def OD_DIF(true_data, noisy_data, contamination): # Deep Isolation Forest (Xu et al., 2023)

    # Step 1: Create and train the model with specified parameters
    model = DIF(contamination = contamination)
    model.fit(true_data)
    
    # Step 2: Predict outliers
    y_pred = model.predict(noisy_data)
    
    # Step 3: Get the predicted labels
    labels = y_pred

    # Step 4: Separate inliers and outliers for plotting
    inliers = noisy_data[labels == 0]
    outliers = noisy_data[labels == 1]

    return outliers, labels