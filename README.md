# sktime-xai: Explainable AI for Time Series

> **Interpretability for the [sktime](https://www.sktime.net) ecosystem** — from SHAP attributions for forecasters to saliency maps for ROCKET classifiers.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org)
[![License: BSD-3](https://img.shields.io/badge/license-BSD--3-green)](LICENSE)

---

## ✨ What's in v0.1

| Module | Feature | Description |
|--------|---------|-------------|
| **Forecasting** | `SHAPForecastExplainer` | SHAP (Kernel / Permutation) for any sktime forecaster |
| **Classification** | `RocketExplainer` | Weight-based kernel attribution for ROCKET classifiers |
| **Classification** | `saliency_map()` | Temporal saliency heatmap — per-timestep importance scores |
| **Visualization** | `plot_temporal_heatmap` | Lag-importance heatmaps for forecasting |
| **Visualization** | `plot_saliency()` | GradCAM-style saliency overlay for time series |

## 📦 Installation

```bash
pip install sktime-xai
```

Or install from source:

```bash
git clone https://github.com/sktime/sktime-xai.git
cd sktime-xai
pip install -e .
```

## ⚡ Quick Start

### Explain a Forecast (SHAP)

```python
from sktime.datasets import load_airline
from sktime.forecasting.trend import TrendForecaster
from sktime_xai.forecasting import SHAPForecastExplainer

y = load_airline()
forecaster = TrendForecaster().fit(y[:-12])

explainer = SHAPForecastExplainer(forecaster)
result = explainer.explain(y[-24:], fh=1)
result.plot()
```

### Explain a ROCKET Classifier

```python
from sktime.datasets import load_gunpoint
from sktime.classification.kernel_based import RocketClassifier
from sktime_xai.classification._rocket_explainer import RocketExplainer, RocketExplanation

X_train, y_train = load_gunpoint(split="train")
X_test, y_test = load_gunpoint(split="test")

clf = RocketClassifier(num_kernels=500, random_state=42)
clf.fit(X_train, y_train)

explainer = RocketExplainer(clf)

# Top-K kernel activations
explanation = explainer.explain(X_test.iloc[[0]], class_index=0, top_k=3)
explanation.plot()

# Temporal saliency map
saliency = explainer.saliency_map(X_test.iloc[[0]], class_index=0)
RocketExplanation.plot_saliency(saliency)
```

## 📓 Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 01 | [Forecasting SHAP](examples/01_forecasting_shap.ipynb) | SHAP explanations for time series forecasters |
| 02 | [ROCKET Explainer](examples/02_rocket_explainer.ipynb) | Kernel activations + temporal saliency maps |

---

## 🗺️ Roadmap to v1.0

### Model-Agnostic Explanations

| Feature | Status | Description |
|---------|--------|-------------|
| SHAP Forecaster | ✅ v0.1 | Kernel/Permutation SHAP for any sktime forecaster |
| SHAP with temporal segment masking | 🔲 v0.3 | Respects autocorrelation — perturbs contiguous segments, not individual timesteps |
| LIME-TS | 🔲 v0.3 | LIME with contiguous segment perturbations for classifiers |
| Perturbation-based lag attribution | 🔲 v0.4 | Occlude past windows to measure forecaster sensitivity |

### Model-Specific (Gradient / Architecture)

| Feature | Status | Description |
|---------|--------|-------------|
| ROCKET kernel attribution | ✅ v0.1 | Linear weight × feature value decomposition |
| ROCKET saliency map | ✅ v0.1 | Per-timestep importance via weighted kernel aggregation |
| GradCAM for InceptionTime / ResNet | 🔲 v0.4 | Gradient-weighted class activation maps for deep TS classifiers |
| Integrated Gradients | 🔲 v0.4 | Attribution via path integration for PyTorch classifiers |
| Attention rollout | 🔲 v0.5 | Aggregate attention maps for Transformer-based models |
| ARIMA component decomposition | 🔲 v0.5 | AR/MA attribution — ante-hoc interpretability for ARIMA family |

### Mechanistic Interpretability

| Feature | Status | Description |
|---------|--------|-------------|
| ROCKET kernel probing | 🔲 v0.4 | Map high-weight kernels to human-interpretable temporal concepts |
| Linear probing on intermediate layers | 🔲 v0.5 | Probe hidden representations in deep TS models |

### Faithfulness Evaluation

| Metric | Status | Description |
|--------|--------|-------------|
| Comprehensiveness | 🔲 v0.6 | Does removing important features change the prediction? |
| Sufficiency | 🔲 v0.6 | Do important features alone reproduce the prediction? |
| Stability | 🔲 v0.6 | Are explanations consistent for similar inputs? |

> **Goal:** Explanations should be *compared and measured*, not just visualised.

---

## 🏗️ Architecture

```
sktime_xai/
├── base/              # BaseExplainer interface
├── forecasting/       # SHAPForecastExplainer
├── classification/    # RocketExplainer (+ future: DeepClassifierExplainer)
└── visualization/     # plot_temporal_heatmap, plot_saliency
```

## 🤝 Contributing

Contributions welcome! See the roadmap above for planned features. Open an issue to discuss your idea before starting a PR.

## 📄 License

BSD-3-Clause
