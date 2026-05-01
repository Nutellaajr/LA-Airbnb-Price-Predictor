# LA Airbnb Price Predictor

This repository includes a Streamlit app for exploring Los Angeles Airbnb listings and predicting nightly prices with the tuned Gradient Boosting model.

## Run the App

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Required Model Files

The app expects these files:

- `feature_and_modeling/final_model.pkl`
- `feature_and_modeling/preprocessor.pkl`
- `feature_and_modeling/feature_importance.csv`
- `feature_and_modeling/data_with_features.csv`

The model predicts `log_price = log(1 + price)`. The app converts predictions back to dollars with `np.expm1()`.

If `feature_and_modeling/final_model.pkl` is missing, open `feature_and_modeling/feature_and_modeling_code.ipynb` and add this line immediately after the cell that creates and evaluates `gb_tuned`, before saving the final model comparison:

```python
joblib.dump(gb_tuned, "feature_and_modeling/final_model.pkl")
```

If you run the notebook from inside the `feature_and_modeling/` folder, use:

```python
joblib.dump(gb_tuned, "final_model.pkl")
```

## App Features

- Airbnb nightly price prediction form
- Same feature engineering logic used in the modeling notebook
- Fitted preprocessor loaded from `feature_and_modeling/preprocessor.pkl`
- Interactive LA map with filters for room type, price, neighbourhood group, and bedrooms
- Top Gradient Boosting feature importances from `feature_and_modeling/feature_importance.csv`
