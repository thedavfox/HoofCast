# Prediction model

This folder contains the final model selected for Section 6: an RBF-kernel
support vector regression (SVR) model using Feature Set B. It predicts
`outcome_activity_not_stationary`, the proportion of the following 100-second
outcome window during which the horse was not stationary.

## Saved model

`selected_rbf_svr_set_b.pkl` is trained on all 1,660 observations in
`Data/exploration_data.csv` (the development 80% split). Its hyperparameters
were selected with shuffled five-fold cross-validation using seed 226 and mean
validation MAE as the primary criterion:

- `C=1.0`
- `gamma=0.01`
- `epsilon=0.01`

The fitted preprocessing is stored inside the pickle: median imputation and
standardization of continuous inputs, plus one-hot encoding of
`pre_posture_dominant`. Raw predictions are clipped to the valid proportion
range, 0 through 1, by the prediction utility.

## Feature Set B

Set B contains these predictors:

- time of day as circular sine and cosine features;
- weather temperature, humidity, precipitation, cloud cover, shortwave
  radiation, wind speed, wind gusts, and day/night status;
- wind direction as circular sine and cosine features;
- pre-period posture, visibility, transition, and secondary-behavior
  proportions; and
- the one-hot-encoded dominant pre-period posture.

The exact raw CSV columns required are listed in
`selected_rbf_svr_set_b_metadata.json` after training.

## Train or recreate the model

From the repository root, with the project Python environment activated:

```bash
python prediction/train_selected_model.py
```

This recreates both the pickle and its metadata file from the development CSV.

## Predict from a CSV

Your input CSV needs the Set B raw predictor columns; it does **not** need an
outcome column. Include `observation_id` if you want it retained in the output.

```bash
python prediction/predict_with_selected_model.py path/to/new_data.csv \
  --output prediction/outputs/new_predictions.csv
```

The output contains `prediction_not_stationary`, the clipped predicted
proportion of a 100-second window spent non-stationary. For example, `0.20`
corresponds to a prediction of approximately 20 non-stationary seconds.
