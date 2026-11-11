"""Feature engineering shared by the notebook and the Streamlit app.

The same function runs inside the saved pipeline (as a FunctionTransformer step),
so a new car only needs the raw inputs: Year, Kilometers_Driven, Engine, Power, ...
"""
import numpy as np
import pandas as pd

# The data was collected in 2019 (newest car: 2019), so ages are counted from 2020.
REF_YEAR = 2020


def add_features(X):
    """Add Age, Km_per_year and Power_per_litre, and drop Year.

    | Feature         | Formula                       | Why                                  |
    |-----------------|-------------------------------|--------------------------------------|
    | Age             | REF_YEAR - Year               | price drops with age, not with year  |
    | Km_per_year     | Kilometers_Driven / Age       | how hard the car was used            |
    | Power_per_litre | Power / (Engine / 1000)       | sporty vs economy engine             |
    """
    X = X.copy()
    X["Age"] = REF_YEAR - X["Year"]
    X["Km_per_year"] = X["Kilometers_Driven"] / X["Age"].clip(lower=1)
    X["Power_per_litre"] = X["Power"] / (X["Engine"] / 1000)
    X = X.replace([np.inf, -np.inf], np.nan)
    return X.drop(columns=["Year"])
