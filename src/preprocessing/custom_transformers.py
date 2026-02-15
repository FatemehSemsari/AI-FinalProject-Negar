import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip numeric features to reduce extreme outliers based on training quantiles."""
    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=float)
        self.lower_ = np.nanquantile(X_arr, self.lower_quantile, axis=0)
        self.upper_ = np.nanquantile(X_arr, self.upper_quantile, axis=0)
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        return np.clip(X_arr, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features) if input_features is not None else None