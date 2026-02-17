# sktime-xai: Explainable AI for Time Series

`sktime-xai` brings interpretability to the `sktime` ecosystem. It provides a unified interface for explaining forecasting and classification models, helping you answer "why" your model made a specific prediction.

## 🚀 Features (MVP 0.1)

- **Unified API**: Compatible with `sktime` forecasters and classifiers.
- **Forecasting Explanation**:
    - `SHAPForecastExplainer`: Apply SHAP (Kernel/Permutation) to any sktime forecaster.
    - Visualize which past time steps contributed most to a future prediction.
- **Visualization**:
    - `plot_temporal_heatmap`: Intuitive heatmaps showing feature importance over time.

## 📦 Installation

```bash
pip install sktime-xai
```

## ⚡ Quick Start: Explaining a Forecast

```python
from sktime.datasets import load_airline
from sktime.forecasting.trend import TrendForecaster
from sktime_xai.forecasting import SHAPForecastExplainer

# 1. Load Data & Train
y = load_airline()
forecaster = TrendForecaster().fit(y[:-12])

# 2. Explain
explainer = SHAPForecastExplainer(forecaster)
# Explain the forecast for the next step, based on the last 24 months
result = explainer.explain(y[-24:], fh=1) 

# 3. Visualize
result.plot()
```

## 🗺️ Roadmap

- **Classification**: LIME for time series, CAM for neural networks.
- **Mechanistic Interpretability**: ROCKET kernel probing, activation patching.
- **Evaluation**: Faithfulness and stability metrics.
