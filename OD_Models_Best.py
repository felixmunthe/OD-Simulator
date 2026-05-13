from pyod.models.gmm import GMM
from pyod.models.mcd import MCD
from pyod.models.lof import LOF
from pyod.models.feature_bagging import FeatureBagging

def OD_MCD(true_data, noisy_data, contamination, return_scores = False):
    model = MCD(contamination = contamination)
    model.fit(true_data)

    labels = model.predict(noisy_data)
    outliers = noisy_data[labels == 1]

    if return_scores:
        scores = model.decision_function(noisy_data)
        threshold = getattr(model, "threshold_", None)
        return outliers, labels, scores, threshold

    return outliers, labels

def OD_LOF(true_data, noisy_data, contamination, neighbor, return_scores = False):
    model = LOF(contamination = contamination, n_neighbors = neighbor)
    model.fit(true_data)

    labels = model.predict(noisy_data)
    outliers = noisy_data[labels == 1]

    if return_scores:
        scores = model.decision_function(noisy_data)
        threshold = getattr(model, "threshold_", None)
        return outliers, labels, scores, threshold

    return outliers, labels

def OD_FB(true_data, noisy_data, estimator, contamination, return_scores = False):
    model = FeatureBagging(n_estimators = estimator, contamination = contamination)
    model.fit(true_data)

    labels = model.predict(noisy_data)
    outliers = noisy_data[labels == 1]

    if return_scores:
        scores = model.decision_function(noisy_data)
        threshold = getattr(model, "threshold_", None)
        return outliers, labels, scores, threshold

    return outliers, labels

def OD_GMM(true_data, noisy_data, component, tolerance, return_scores = False):
    model = GMM(n_components = component, tol = tolerance)
    model.fit(true_data)

    labels = model.predict(noisy_data)
    outliers = noisy_data[labels == 1]

    if return_scores:
        scores = model.decision_function(noisy_data)
        threshold = getattr(model, "threshold_", None)
        return outliers, labels, scores, threshold

    return outliers, labels