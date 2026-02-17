
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sktime.datasets import load_airline
from sktime.forecasting.trend import TrendForecaster

from sktime_xai.forecasting._shap_forecaster import SHAPForecastExplainer

print("Loading data...")
y = load_airline()
y_train = y[:-12]
y_test = y[-12:]

print(f"Train shape: {y_train.shape}, Test shape: {y_test.shape}")

# Simple trend forecaster
print("Fitting forecaster...")
forecaster = TrendForecaster()
forecaster.fit(y_train)

# The context window to explain
y_context = y_train[-24:]

# initialize explainer
print("Initializing SHAP explainer...")
explainer = SHAPForecastExplainer(forecaster)
explainer.fit(y_train) # Initialize background

# Explain prediction at fh=1 (next step)
print("Running explain()...")
result = explainer.explain(y_context, fh=1)

print("Explanation complete!")
print("Attribution scores (head):")
print(result.attribution_values.head())

# Plot (headless server, so we just check object creation)
fig = result.plot()
print("Plot created successfully.")
print("MVP verified!")
