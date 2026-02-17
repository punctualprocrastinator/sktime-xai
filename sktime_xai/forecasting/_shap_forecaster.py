"""SHAP explainer for sktime forecasters."""

import numpy as np
import pandas as pd
import shap
from sktime.forecasting.base import BaseForecaster

from sktime_xai.base._base_explainer import BaseExplainer
from sktime_xai.base._base_attribution import AttributionResult



from sklearn.base import clone

class SHAPForecastExplainer(BaseExplainer):
    """SHAP-based explainer for forecasting models.
    
    Uses shap.KernelExplainer or similar to explain forecasting predictions
    based on past lag windows.
    """

    def __init__(self, forecaster, background_samples=50):
        """
        Parameters
        ----------
        forecaster : fitted sktime forecaster
            The model to explain.
        background_samples : int
            Number of background samples to use for SHAP (if applicable).
        """
        super().__init__()
        self.forecaster = forecaster
        self.background_samples = background_samples
        self._shap_explainer = None
        self._fitted_fh = None
        
    def fit(self, X, y=None):
        """Initialize the SHAP explainer with background data.
        
        For forecasting, 'X' here is usually the historical training data (y_train).
        We will create valid windows from validation data or simple lags if X is passed.
        
        If X is a Series/DataFrame (the time series y), we treat it as the background
        distribution for the "features" (lags).
        """
        # Note: This is a simplified "Speedrun" implementation.
        # In a full implementation, we'd inspect the forecaster's internal window length.
        
        # Assumption: Model is already fitted.
        # if not self.forecaster.is_fitted:
        #      raise ValueError("Forecaster must be fitted before initialization.")
        
        # We need to define a prediction function for SHAP that takes "tabular" data
        # and outputs the forecast.
        # Tabular data = [n_samples, window_length]
        
        # For this MVP, we will use a masker based on median/mean of X
        # And we assume univariate mainly for the demo.
        self._X_background = X  
        return self

    def explain(self, y, fh=1, **kwargs):
        """Explain a forecast.
        
        Parameters
        ----------
        y : pd.Series
            The historical data leading up to the forecast point. 
            Must be long enough to cover the model's required lookback window.
        fh : int
            Forecasting horizon step to explain (relative to end of y).
            Default is 1 (next step).
            
        Returns
        -------
        AttributionResult
        """
        # 1. Prepare function f(x) -> prediction
        # SHAP passes a numpy array of shape (n_samples, n_features)
        # We need to convert that back to a Series and feed to forecaster
        
        # Need to know window length. 
        # For general models, this is hard. For MVP, let's assume valid input `y` 
        # is the exact window needed.
        
        # Wrapped predict function for SHAP
        def predict_fn(window_array):
            # window_array: (n_samples, window_len)
            preds = []
            for row in window_array:
                # Reconstruct series
                # We need context. For now, we assume row is the *recent history*.
                # We construct a temporary series structure.
                # sktime often needs an index.
                # We'll use a dummy index relative to end of y for prediction?
                # Actually, sktime predict(fh, y=history)
                
                # Careful: 'row' is a numpy array of values.
                # We make a pd.Series with integer index 0..len-1
                y_hist = pd.Series(row)
                
                # Predict
                # Strategy: Clone and Refit on the history
                # This treats the logical task f(history) -> forecast
                # properly for "Attribution of History".
                # It is slow but correct for general estimators.
                
                # Clone
                model_clone = clone(self.forecaster)
                
                # Fit on the perturbed history
                # We assume y_hist is the Training data for this instance
                model_clone.fit(y=y_hist)
                
                # Predict
                pred = model_clone.predict(fh=[fh])
                preds.append(pred.iloc[0])
            
            return np.array(preds)


        # 2. Setup SHAP
        # We use the provided 'y' as the instance to explain.
        # 'y' should be the recent history window.
        X_instance = y.values.reshape(1, -1)
        
        # Background: we need a background dataset. 
        # Ideally passed in fit(). If not, we summarize y itself or zeros.
        # Using a zero baseline or mean baseline is common.
        background = np.zeros((1, X_instance.shape[1])) + np.mean(X_instance)
        
        # Initialize KernelExplainer (agnostic)
        explainer = shap.KernelExplainer(predict_fn, background)
        
        # 3. Calculate SHAP values
        shap_values = explainer.shap_values(X_instance, nsamples=100)
        
        # 4. Package results
        # shap_values is (1, n_features) (or list if output is vector, but here output is scalar)
        
        # If output is scalar
        if isinstance(shap_values, list):
             attr = shap_values[0]
        else:
             attr = shap_values
             
        attr_series = pd.Series(attr.flatten(), index=y.index)
        
        return AttributionResult(
            attribution_values=attr_series, 
            X_explained=y, 
            method_name="SHAP"
        )
