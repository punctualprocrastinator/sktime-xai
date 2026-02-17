"""ROCKET Explainer for sktime ROCKET classifiers."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sktime.base import BaseObject
from sklearn.pipeline import Pipeline
from sktime.classification.kernel_based import RocketClassifier
from sktime.transformations.panel.rocket import Rocket

class RocketExplainer(BaseObject):
    """Explainer for ROCKET models.
    
    Interpretability for Random Convolutional Kernel Transform (ROCKET).
    It works by:
    1. Extracting linear weights from the downstream classifier (e.g. Ridge).
    2. Identifying the most important features (PPV or Max pooling of specific kernels).
    3. Reconstructing the kernels associated with those features.
    4. Visualizing where those kernels "fire" (activate) on the input time series.
    """
    
    def __init__(self, classifier):
        """
        Parameters
        ----------
        classifier : RocketClassifier or Pipeline
            Fitted model containing a Rocket transformer and a linear classifier.
        """
        super().__init__()
        self.classifier = classifier
        self._extract_components()
        
    def _extract_components(self):
        """Extract transformer and linear classifier from the wrapped model.

        For RocketClassifier the fitted components are stored as:
        - ``clf.estimator_.classifier_``       → fitted RidgeClassifierCV (has ``coef_``)
        - ``clf.estimator_.transformers_``      → fitted TransformerPipeline (can ``.transform()``)
        - ``clf.estimator_.transformers_.steps_[0][1]`` → fitted Rocket object (has ``kernels``)
        """
        self.transformer = None          # fitted TransformerPipeline (for .transform())
        self.linear_model = None          # fitted classifier (has .coef_)
        self._rocket_object = None        # raw Rocket (has .kernels for visualization)

        # 1. sklearn Pipeline case
        if isinstance(self.classifier, Pipeline):
            self._extract_from_pipeline(self.classifier)
            return

        # 2. RocketClassifier case
        if isinstance(self.classifier, RocketClassifier):
            est = getattr(self.classifier, "estimator_", None)
            if est is None:
                raise ValueError("RocketClassifier must be fitted first.")

            # Linear model  (estimator_.classifier_)
            clf_ = getattr(est, "classifier_", None)
            if clf_ is not None and hasattr(clf_, "coef_"):
                self.linear_model = clf_

            # Transformer pipeline  (estimator_.transformers_)
            tp = getattr(est, "transformers_", None)
            if tp is not None and getattr(tp, "_is_fitted", False):
                self.transformer = tp

                # Rocket object for kernel visualization  (steps_[0][1])
                steps_ = getattr(tp, "steps_", [])
                for item in steps_:
                    if isinstance(item, tuple) and len(item) >= 2:
                        obj = item[1]
                        if "Rocket" in type(obj).__name__ and hasattr(obj, "kernels"):
                            self._rocket_object = obj
                            break

            if self.transformer is not None and self.linear_model is not None:
                return

            raise ValueError(
                "RocketClassifier must be fitted "
                "(could not find fitted transformer and/or classifier)."
            )

        raise ValueError(
            "Unsupported model type. Pass RocketClassifier or "
            "Pipeline(Rocket, clf)."
        )

    def _extract_from_pipeline(self, pipeline):
        """Helper to extract from a pipeline (sklearn or sktime)."""
        self.transformer = None
        self.linear_model = None
        
        steps = getattr(pipeline, "steps", getattr(pipeline, "_steps", None))
        if steps is None:
             raise ValueError("Pipeline does not have 'steps' or '_steps' attribute.")

        for name, step in steps:
            # Check for Rocket transformer
            if isinstance(step, Rocket) or hasattr(step, "kernels") or "rocket" in name.lower():
                self.transformer = step
            
            # Check for classifier (MUST have coef_)
            if hasattr(step, "coef_"):
                self.linear_model = step
            elif hasattr(step, "best_estimator_") and hasattr(step.best_estimator_, "coef_"):
                self.linear_model = step.best_estimator_
        
        if self.transformer is None:
            raise ValueError("Could not find Rocket transformer in Pipeline.")
        if self.linear_model is None:
            raise ValueError("Could not find linear classifier in Pipeline (must have coef_).")


    def _get_target_coefs(self, class_index=0):
        """Get linear model coefficients for a given class."""
        coefs = self.linear_model.coef_
        if coefs.ndim == 1:
            return coefs if class_index == 1 else -coefs
        return coefs[class_index]

    def _prepare_X(self, X):
        """Ensure X is a single-row DataFrame."""
        if isinstance(X, pd.Series):
            return pd.DataFrame([X])
        return X

    @staticmethod
    def _convolve_kernel(x_data, w, d, b):
        """Compute dilated convolution activation for one kernel."""
        n_time = len(x_data)
        w_len = len(w)
        eff_len = (w_len - 1) * d + 1
        out_len = n_time - eff_len + 1
        if out_len <= 0:
            return np.zeros(n_time)
        activation = np.full(n_time, np.nan)
        for t in range(out_len):
            dot = 0.0
            for j in range(w_len):
                dot += x_data[t + j * d] * w[j]
            activation[t] = dot + b
        return activation

    @staticmethod
    def _extract_x_data(X_instance):
        """Pull raw 1-D numpy array from a sktime row."""
        if isinstance(X_instance, pd.Series) and isinstance(
            X_instance.iloc[0], (pd.Series, np.ndarray)
        ):
            x = X_instance.iloc[0]
            return x.values if isinstance(x, pd.Series) else x
        return X_instance.values.flatten()

    # -----------------------------------------------------------------
    def explain(self, X, class_index=0, top_k=3):
        """Explain classification with top-k kernel contributions.

        Parameters
        ----------
        X : pd.Series or pd.DataFrame (single instance)
        class_index : int
        top_k : int

        Returns
        -------
        RocketExplanation
        """
        target_coefs = self._get_target_coefs(class_index)
        n_kernels = self._rocket_object.num_kernels

        X_df = self._prepare_X(X)
        X_trans = self.transformer.transform(X_df)
        X_trans_val = X_trans.iloc[0].values

        contributions = target_coefs * X_trans_val
        top_indices = np.argsort(contributions)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if idx < n_kernels:
                kernel_idx, feat_type = idx, "PPV"
            else:
                kernel_idx, feat_type = idx - n_kernels, "Max"
            results.append({
                "kernel_idx": kernel_idx,
                "type": feat_type,
                "contribution": contributions[idx],
                "weight": target_coefs[idx],
                "value": X_trans_val[idx],
            })

        return RocketExplanation(
            self._rocket_object, X_df.iloc[0], results,
        )

    # -----------------------------------------------------------------
    def saliency_map(self, X, class_index=0):
        """Compute temporal saliency map (importance per timestep).

        Aggregates *all* kernel activations weighted by their linear
        model coefficients.  The result is a 1-D array, one value per
        timestep, showing how much each timestep contributed to the
        predicted class.

        Parameters
        ----------
        X : pd.Series or pd.DataFrame (single instance)
        class_index : int

        Returns
        -------
        dict with keys:
            x_data   – raw time series values (1-D np array)
            saliency – importance score per timestep (1-D np array)
        """
        target_coefs = self._get_target_coefs(class_index)
        n_kernels = self._rocket_object.num_kernels

        X_df = self._prepare_X(X)
        X_trans = self.transformer.transform(X_df)
        X_trans_val = X_trans.iloc[0].values

        x_data = self._extract_x_data(X_df.iloc[0])
        n_time = len(x_data)
        saliency = np.zeros(n_time)

        kernels = self._rocket_object.kernels
        weights, lengths, biases, dilations, paddings = kernels[:5]

        for k in range(n_kernels):
            w_start = int(np.sum(lengths[:k]))
            w_len = int(lengths[k])
            w = weights[w_start: w_start + w_len]
            b = biases[k]
            d = int(dilations[k])

            act = self._convolve_kernel(x_data, w, d, b)

            # Two features per kernel: PPV (index k) and Max (index k + n_kernels)
            ppv_weight = target_coefs[k]           * X_trans_val[k]
            max_weight = target_coefs[k + n_kernels] * X_trans_val[k + n_kernels]
            combined = ppv_weight + max_weight

            # Spread contribution across the timesteps the kernel touches
            valid = ~np.isnan(act)
            if valid.any():
                saliency[valid] += combined * np.abs(act[valid])

        return {"x_data": x_data, "saliency": saliency}

class RocketExplanation:
    """Container for a ROCKET explanation with plotting helpers."""

    def __init__(self, rocket_object, X_instance, top_kernels_info):
        self.rocket = rocket_object
        self.X_instance = X_instance
        self.top_kernels_info = top_kernels_info

    # -----------------------------------------------------------------
    def plot(self):
        """Plot each top-k kernel's activation overlaid on the time series."""
        n = len(self.top_kernels_info)
        fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)
        if n == 1:
            axes = [axes]

        x_data = RocketExplainer._extract_x_data(self.X_instance)
        kernels = self.rocket.kernels
        weights, lengths, biases, dilations, paddings = kernels[:5]

        for i, info in enumerate(self.top_kernels_info):
            ax = axes[i]
            k_idx = info["kernel_idx"]

            w_start = int(np.sum(lengths[:k_idx]))
            w_len = int(lengths[k_idx])
            w = weights[w_start: w_start + w_len]
            b, d = biases[k_idx], int(dilations[k_idx])

            activation = RocketExplainer._convolve_kernel(x_data, w, d, b)

            # Signal
            ax.plot(x_data, color="gray", alpha=0.5, label="Signal")

            # Activation on twin axis
            ax2 = ax.twinx()
            ax2.plot(activation, color="red", linewidth=1.5,
                     label=f"Kernel {k_idx}")

            if info["type"] == "Max":
                max_t = np.nanargmax(activation)
                ax2.scatter([max_t], [activation[max_t]],
                            color="black", marker="*", s=100,
                            label="Max Value")

            if info["type"] == "PPV":
                ax2.axhline(0, color="blue", ls="--", alpha=0.5)

            ax.set_title(
                f"Rank {i+1}: Kernel {k_idx} ({info['type']}) "
                f"| Contrib: {info['contribution']:.2f} | Dilation: {d}"
            )

        plt.tight_layout()
        return fig

    # -----------------------------------------------------------------
    @staticmethod
    def plot_saliency(saliency_result, ax=None, cmap="YlOrRd"):
        """Plot temporal saliency heatmap.

        Parameters
        ----------
        saliency_result : dict  (output of ``RocketExplainer.saliency_map``)
        ax : matplotlib Axes, optional
        cmap : str
        """
        x_data = saliency_result["x_data"]
        sal = saliency_result["saliency"]
        # Normalise to [0, 1]
        sal_abs = np.abs(sal)
        sal_max = sal_abs.max() or 1.0
        sal_norm = sal_abs / sal_max

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 3))
        else:
            fig = ax.figure

        t = np.arange(len(x_data))

        # Background heatmap as coloured bars
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        norm = Normalize(vmin=0, vmax=1)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        for i in range(len(t) - 1):
            ax.axvspan(t[i], t[i + 1], color=sm.to_rgba(sal_norm[i]),
                       alpha=0.85)

        ax.plot(t, x_data, color="black", linewidth=1.2, label="Signal")
        cb = fig.colorbar(sm, ax=ax, pad=0.02)
        cb.set_label("Importance")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Value")
        ax.set_title("Temporal Saliency Map")
        ax.set_xlim(t[0], t[-1])
        plt.tight_layout()
        return fig
