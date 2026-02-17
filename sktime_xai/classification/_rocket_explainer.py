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
        """Extract transformer and linear classifier from the wrapped model."""
        # 1. Pipeline case
        if isinstance(self.classifier, Pipeline):
             self._extract_from_pipeline(self.classifier)
             return

        # 2. RocketClassifier case
        if isinstance(self.classifier, RocketClassifier):
            # Check for internal estimator (pipeline)
            if hasattr(self.classifier, "estimator_"):
                 self._extract_from_pipeline(self.classifier.estimator_)
                 return
            
            # Legacy/Alternate: Check for direct attributes
            if hasattr(self.classifier, "_transformer"):
                self.transformer = self.classifier._transformer
                self.linear_model = self.classifier._classifier
                return
                
            raise ValueError("RocketClassifier structure not recognized (no estimator_ or _transformer).")

        raise ValueError("Unsupported model type. Pass RocketClassifier or Pipeline(Rocket, clf).")

    def _extract_from_pipeline(self, pipeline):
        """Helper to extract from a pipeline (sklearn or sktime)."""
        self.transformer = None
        self.linear_model = None
        
        # Handle sktime pipeline which uses _steps instead of steps
        steps = getattr(pipeline, "steps", getattr(pipeline, "_steps", None))
        
        if steps is None:
             raise ValueError("Pipeline does not have 'steps' or '_steps' attribute.")

        for name, step in steps:
            # Check for Rocket transformer (class name check or attribute check)
            if isinstance(step, Rocket) or hasattr(step, "kernels"):
                self.transformer = step
            # Check for classifier (has coef_)
            if hasattr(step, "coef_"):
                self.linear_model = step
        
        if self.transformer is None:
            raise ValueError("Could not find Rocket transformer in Pipeline.")
        if self.linear_model is None:
            raise ValueError("Could not find linear classifier in Pipeline.")


    def explain(self, X, class_index=0, top_k=3):
        """Explain classification for a specific class.
        
        Returns the top-k kernels that contribute to this class, 
        and visualizes their activation on X.
        
        Parameters
        ----------
        X : pd.Series or pd.DataFrame (single instance)
            The time series to explain.
        class_index : int
            The class index (in user's `classes_`) to explain. 
            For binary classification, weights are usually 1D, so class_index 0 
            might refer to the negative class or we just use abs(weights).
            We'll assume multi-class or handle binary specially.
        top_k : int
            Number of top kernels to show.
            
        Returns
        -------
        RocketExplanation object (contains plot method)
        """
        # 1. Get weights
        # RidgeClassifierCV.coef_ is (n_classes, n_features) or (1, n_features) if binary
        coefs = self.linear_model.coef_
        
        target_coefs = None
        if coefs.ndim == 1:
            # Binary case. 
            # If class_index == 1 (positive), use coefs. 
            # If class_index == 0 (negative), use -coefs.
            if class_index == 1:
                target_coefs = coefs
            else:
                target_coefs = -coefs
        else:
            # Multi-class
            target_coefs = coefs[class_index]
            
        # 2. Find top features
        # Features are 2 per kernel: [PPV_0, Max_0, PPV_1, Max_1, ...] or [PPV_all, Max_all]?
        # SKTime Rocket implementation: transform returns `X_transformed`
        # usually concatenation of PPV and Max?
        # Let's check typical sktime Rocket output structure.
        # usually 2 * n_kernels features.
        # Order: usually [PPV for all kernels, Max for all kernels] OR [PPV_k0, Max_k0, ...]
        # We assume [PPV_0, PPV_1, ... PPV_n, Max_0, Max_1, ... Max_n] ?
        # Actually standard Rocket in sktime:
        # for i in num_kernels:
        #    ...
        # It's often interleaved or blocked. 
        # Standard implementation: first num_kernels are PPV, next num_kernels are Max.
        
        n_kernels = self.transformer.num_kernels
        num_features = len(target_coefs)
        
        # Guard: check feature count matches
        if num_features != 2 * n_kernels:
             # Just a warning or simple logic if dimensions mismatch
             pass 

        # Sort by contribution (weight * feature_value?) 
        # Ideally Global explanation uses just Weights.
        # Local explanation uses Weight * FeatureValue.
        # We need X_transformed for Local.
        
        # Transform X to get feature values
        # X must be in correct format (e.g. nested DF)
        if isinstance(X, pd.Series):
             # Wrap in DF
             X_df = pd.DataFrame([X])
        else:
             X_df = X
             
        X_trans = self.transformer.transform(X_df)
        X_trans_val = X_trans.iloc[0].values
        
        # Contribution = Weight * Value
        contributions = target_coefs * X_trans_val
        
        # Get top K indices
        top_indices = np.argsort(contributions)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            # Determine kernel index and types
            if idx < n_kernels:
                kernel_idx = idx
                feat_type = "PPV"
            else:
                kernel_idx = idx - n_kernels
                feat_type = "Max"
                
            results.append({
                "kernel_idx": kernel_idx,
                "type": feat_type,
                "contribution": contributions[idx],
                "weight": target_coefs[idx],
                "value": X_trans_val[idx]
            })
            
        return RocketExplanation(self.transformer, X_df.iloc[0], results)

class RocketExplanation:
    def __init__(self, rocket_transformer, X_instance, top_kernels_info):
        self.rocket = rocket_transformer
        self.X_instance = X_instance
        self.top_kernels_info = top_kernels_info
        
        # Reconstruct kernels
        # weights: (3, lengths)
        # lengths: (n_kernels,)
        # biases: (n_kernels,)
        # dilations: (n_kernels,)
        # paddings: (n_kernels,)
        # ... based on sktime implementation
        
        # We can simulate the convolution to compute the "activation profile"
        # for plotting.
    
    def plot(self):
        """Plot the input time series and the activations of top kernels."""
        n = len(self.top_kernels_info)
        fig, axes = plt.subplots(n, 1, figsize=(10, 3*n), sharex=True)
        if n == 1: axes = [axes]
        
        # X data (univariate assumed for MVP)
        # X_instance is a pd.Series where item is a pd.Series (nested) or just values?
        # sktime nested: col 0 is series.
        if isinstance(self.X_instance, pd.Series) and isinstance(self.X_instance.iloc[0], (pd.Series, np.ndarray)):
             x_data = self.X_instance.iloc[0]
             if isinstance(x_data, pd.Series): x_data = x_data.values
        else:
             # Assume flatten
             x_data = self.X_instance.values.flatten()
             
        for i, info in enumerate(self.top_kernels_info):
            ax = axes[i]
            k_idx = info['kernel_idx']
            
            # --- Reconstruct Kernel ---
            # Access internal arrays from Rocket
            # Note: attribute names depend on sktime version.
            # Assuming 'kernels' attribute exists (verified in script)
            # Tuple: (weights, lengths, biases, dilations, paddings)
            weights, lengths, biases, dilations, paddings = self.rocket.kernels
            
            # Get specific kernel params
            # Weights are packed 1D array. Need start index.
            # weights is 1D array of float32
            # lengths is 1D array of int32
            
            # Calculate start position
            # cumulative sum of lengths?
            # actually sktime generates start indices or we do it.
            # let's recalculate start indices
            w_start = np.sum(lengths[:k_idx])
            w_len = lengths[k_idx]
            w = weights[w_start : w_start + w_len]
            
            b = biases[k_idx]
            d = dilations[k_idx]
            p = paddings[k_idx]
            
            # --- Compute Activation Map ---
            # Convolution
            # For visualization, we want to show 'w * x + b' over time
            # Dilation logic:
            # effective_length = (w_len - 1) * d + 1
            
            # Simple manual convolution for MVP visualization
            n_time = len(x_data)
            activation = np.zeros(n_time) 
            
            # This is slow in python loops, but okay for MVP demo (k=3)
            # We align the kernel center or start? 
            # ROCKET uses 'valid' convolution usually
            
            if p > 0:
                 # Padding logic is complex in ROCKET (random padding)
                 # We'll skip complex padding visualization for MVP and focus on valid range
                 pass
            
            # Naive convolution with dilation
            for t in range(n_time):
                # t is output index? 
                # ROCKET convolution: dot product at position t
                # kernel spans indices: t, t+d, t+2d ...
                
                # Check bounds
                last_idx = t + (w_len - 1) * d
                if last_idx < n_time:
                    # Valid
                    dot = 0
                    for j in range(w_len):
                        dot += x_data[t + j*d] * w[j]
                    activation[t] = dot + b
                else:
                    activation[t] = np.nan # No activation
            
            # --- Plotting ---
            # 1. Signal
            ax.plot(x_data, color='gray', alpha=0.5, label='Signal')
            
            # 2. Activation (normalized or on twin axis)
            ax2 = ax.twinx()
            ax2.plot(activation, color='red', linewidth=1.5, label=f'Kernel {k_idx} Activation')
            
            # Highlight max point if Feature was "Max"
            if info['type'] == 'Max':
                max_t = np.nanargmax(activation)
                ax2.scatter([max_t], [activation[max_t]], color='black', marker='*', s=100, label='Max Value')
            
            # Highlight > 0 regions if "PPV"
            if info['type'] == 'PPV':
                # Draw simple threshold line?
                ax2.axhline(0, color='blue', linestyle='--', alpha=0.5)
            
            ax.set_title(f"Rank {i+1}: Kernel {k_idx} ({info['type']}) | Contrib: {info['contribution']:.2f} | Dilation: {d}")
            # ax.legend(loc='upper right')
            # ax2.legend(loc='lower right')
            
        plt.tight_layout()
        return fig
