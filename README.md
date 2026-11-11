# 🚗 Used Car Price Prediction

Predict the price of a used car in India from its details, and try it in a Streamlit app.
Project 1 of the **Zero Grad** Machine Learning course: [zero-grad.com/courses/ml/car-price](https://zero-grad.com/courses/ml/car-price/).

## What's inside

| Path | What it is |
|---|---|
| `notebooks/car_price.ipynb` | The whole project, step by step: diagnosis → cleaning → feature engineering → pipeline → models → tuning → evaluation → saved model |
| `car_price/features.py` | Feature engineering (`Age`, `Km_per_year`, `Power_per_litre`), used inside the saved pipeline |
| `car_price/outlier_handler.py` | `OutlierClipper`: a scikit-learn transformer that learns IQR bounds on the training data and clips to them |
| `app.py` | The Streamlit app |
| `models/car_price_model.joblib` | The trained pipeline (raw car columns in, price in lakh out) |
| `models/app_options.json` | Dropdown choices and input ranges for the app |
| `models/report.json` | Scores and test predictions, used by the project page |
| `data/train-data.csv` | Raw data: [Used Cars Price Prediction](https://www.kaggle.com/datasets/avikasliwal/used-cars-price-prediction) (Kaggle) |

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

To retrain, run the notebook from the `notebooks/` folder (it writes everything in `models/`).

## The model

Ridge regression on polynomial features of the numeric columns, binary-encoded categories and an ordinal `Owner_Type`,
trained on `log(1 + price)` (`TransformedTargetRegressor`), with the polynomial degree and alpha tuned by 5-fold
`GridSearchCV`. Scores are in the notebook and in `models/report.json`.

`Price` is in **lakh** (1 lakh = ₹100,000). Ages are counted from 2020 (the newest car in the data is from 2019),
so the app accepts cars up to 2019.
