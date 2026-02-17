Great idea — there's a real gap here. sktime has rich model variety but almost no unified interpretability layer. Here's a full project blueprint:

---

## 🔬 `sktime-xai` — XAI & Mechanistic Interpretability for Time Series

---

### 🧠 The Core Problem

sktime supports forecasting, classification, clustering, anomaly detection — but answers **none** of the "why" questions:
- *Why did the forecaster spike here?*
- *Which timesteps drove the classification decision?*
- *What internal representation did ROCKET learn?*
- *Why did the anomaly detector flag this point?*

---

### 🏗️ Architecture — Follow sktime's Patterns

sktime uses a clean `BaseEstimator` + tag system. Mirror it:

```
sktime_xai/
├── base/
│   ├── _base_explainer.py       # BaseExplainer(BaseObject)
│   └── _base_attribution.py     # BaseAttribution (output container)
├── forecasting/
│   ├── _shap_forecaster.py      # SHAPForecastExplainer
│   ├── _attention.py            # AttentionRollout (for Transformer-based)
│   └── _perturbation.py        # PerturbationExplainer
├── classification/
│   ├── _shapelet_viz.py         # ShapeletExplainer
│   ├── _lime_ts.py              # LIMETimeSeriesExplainer
│   ├── _cam.py                  # ClassActivationMap (for InceptionTime etc.)
│   └── _rocket_probe.py        # ROCKET kernel probing
├── transformations/
│   └── _feature_importance.py  # tsfresh/catch22 feature attribution
├── detection/
│   └── _anomaly_attribution.py # Why was THIS point anomalous?
├── mechanistic/
│   ├── _activation_patch.py    # Activation patching for TS models
│   ├── _probing.py             # Linear probing on intermediate layers
│   └── _concept_bottleneck.py  # Temporal concept extraction
├── visualization/
│   ├── _temporal_heatmap.py    # Saliency over time axis
│   ├── _waterfall.py           # SHAP-style waterfall for forecasts
│   └── _multivar_plot.py       # Multi-channel attribution
└── metrics/
    ├── _faithfulness.py        # Is the explanation actually faithful?
    ├── _stability.py           # Do similar inputs give similar explanations?
    └── _complexity.py          # Explanation complexity score
```

---

### 🎯 Task-by-Task Explainability Plan

**1. Forecasting Explanations**
```python
from sktime_xai.forecasting import SHAPForecastExplainer

explainer = SHAPForecastExplainer(forecaster=fitted_model, window=24)
explanation = explainer.explain(y_history)

# Returns: AttributionResult with .plot(), .timestep_scores, .feature_scores
explanation.plot(kind="waterfall")  # shows contribution of each lag
```
Methods: SHAP with sliding window background, integrated gradients for neural forecasters, component decomposition for ARIMA (trend/seasonal/residual attribution).

**2. Classification Explanations**
```python
from sktime_xai.classification import LIMETimeSeriesExplainer, CAMExplainer

# Model-agnostic
explainer = LIMETimeSeriesExplainer(classifier=tsf_clf)
exp = explainer.explain(X_test.iloc[[0]])
exp.plot(kind="temporal_heatmap")  # highlights important time regions

# Model-specific (for InceptionTime, ResNet, etc.)
cam = CAMExplainer(classifier=inception_model)
cam.explain(X_test.iloc[[0]]).plot()
```

**3. ROCKET Mechanistic Interpretability** ← most novel contribution
```python
from sktime_xai.mechanistic import ROCKETProber

prober = ROCKETProber(rocket_classifier)
# Which kernels fire for which class?
prober.analyze_kernel_class_alignment()
# Temporal receptive field visualization
prober.plot_kernel_activations(X_test.iloc[[0]])
```

**4. Anomaly Detection Attribution**
```python
from sktime_xai.detection import AnomalyAttributionExplainer

exp = AnomalyAttributionExplainer(detector=fitted_detector)
result = exp.explain(time_series_with_anomaly)
result.plot()  # shows which channels/lags caused the anomaly score
```

---

### 🔩 `BaseExplainer` Design

```python
class BaseExplainer(BaseObject):
    """Base class for all sktime XAI explainers."""

    _tags = {
        "explainer_type": None,        # "local", "global", or "both"
        "requires_fit": True,
        "supports_multivariate": False,
        "task": None,                  # "forecasting", "classification", etc.
        "output_type": "AttributionResult",
    }

    def fit(self, estimator, X=None, y=None):
        """Fit the explainer to a model and optional background data."""

    def explain(self, X) -> AttributionResult:
        """Generate explanation for input X."""

    def explain_global(self, X, y=None) -> GlobalAttributionResult:
        """Generate dataset-level explanation."""
```

---

### 🧪 Faithfulness Metrics (the hard part most XAI projects skip)

```python
from sktime_xai.metrics import FaithfulnessEvaluator

eval = FaithfulnessEvaluator(explainer, model, X_test)

# Comprehensiveness: does masking top-k timesteps hurt performance?
eval.comprehensiveness(k=5)

# Sufficiency: is top-k alone enough?
eval.sufficiency(k=5)

# Stability: do nearby inputs get similar explanations?
eval.stability(noise_std=0.01, n_trials=20)
```

---

### 🗺️ Development Phases

| Phase | Deliverable | Effort |
|---|---|---|
| **1 — Foundation** | `BaseExplainer`, `AttributionResult`, perturbation explainer for forecasting | 2-3 weeks |
| **2 — Core Methods** | SHAP wrapper for classifiers + forecasters, LIME-TS, temporal heatmaps | 3-4 weeks |
| **3 — Mechanistic** | ROCKET probing, CAM for neural, activation patching | 3-4 weeks |
| **4 — Metrics** | Faithfulness, stability, complexity benchmarks | 2 weeks |
| **5 — Polish** | sktime PR or companion package on PyPI, tutorials | ongoing |

---

### 💡 What Makes This Novel

Most XAI work targets tabular or images. Time series has unique challenges that make this genuinely research-worthy:

- **Temporal dependencies** — masking isn't iid, you need coherent counterfactuals
- **Multi-resolution** — explanations at tick, segment, and trend levels
- **ROCKET kernels** — ~20k random convolutions; nobody's really mapped them to concepts
- **Forecasting horizon attribution** — which past window contributed to which future step?

---

### 🚀 Quick Start to Validate the Idea

Before building the full framework, prototype this to test demand:

```python
# Proof of concept: SHAP for TimeSeriesForestClassifier
import shap
from sktime.classification.interval_based import TimeSeriesForestClassifier

clf = TimeSeriesForestClassifier()
clf.fit(X_train, y_train)

# Wrap predict_proba to work with SHAP
def predict_fn(X_flat):
    X_ts = pd.DataFrame(...)  # reshape back to sktime format
    return clf.predict_proba(X_ts)

explainer = shap.KernelExplainer(predict_fn, X_background_flat)
shap_values = explainer.shap_values(X_test_flat)
# Map back to time axis → temporal saliency map
```
