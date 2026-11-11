import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class OutlierClipper(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Clip (winsorize) numeric columns to bounds learned from the training data.

    method : 'iqr'    -> bounds = Q1 - factor*IQR , Q3 + factor*IQR   (default factor 1.5)
             'zscore' -> bounds = mean - factor*std, mean + factor*std (default factor 3)

    Missing values are allowed and kept as they are (impute before or after).
    Use .set_output(transform='pandas') to get a DataFrame back.
    """

    def __init__(self, method='iqr', factor=None):
        self.method = method
        self.factor = factor

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    def fit(self, X, y=None):
        X = validate_data(self, X, ensure_all_finite='allow-nan', dtype='numeric')

        if self.method == 'iqr':
            factor = 1.5 if self.factor is None else self.factor
            q1, q3 = np.nanpercentile(X, [25, 75], axis=0)
            iqr = q3 - q1
            self.lower_ = q1 - factor * iqr
            self.upper_ = q3 + factor * iqr
        elif self.method == 'zscore':
            factor = 3.0 if self.factor is None else self.factor
            mean, std = np.nanmean(X, axis=0), np.nanstd(X, axis=0)
            self.lower_ = mean - factor * std
            self.upper_ = mean + factor * std
        else:
            raise ValueError("method must be 'iqr' or 'zscore'")

        return self

    def transform(self, X):
        # Learned rule: the same bounds are applied to train, test and any new data
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, ensure_all_finite='allow-nan', dtype='numeric')
        return np.clip(X, self.lower_, self.upper_)
