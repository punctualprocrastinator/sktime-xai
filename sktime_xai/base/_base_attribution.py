"""Container for attribution results."""

import pandas as pd
from sktime.base import BaseObject

from sktime_xai.visualization._temporal_heatmap import plot_temporal_heatmap


class AttributionResult(BaseObject):
    """Container for holding explanation results."""

    def __init__(self, attribution_values, X_explained=None, method_name="Approximation"):
        """Initialize result container.
        
        Parameters
        ----------
        attribution_values : pd.DataFrame or pd.Series
            Attribution scores. Index should match X_explained time index if possible.
        X_explained : pd.DataFrame or pd.Series, optional
            The data that was explained.
        method_name : str
            Name of the explainer method (e.g. "SHAP", "LIME").
        """
        self.attribution_values = attribution_values
        self.X_explained = X_explained
        self.method_name = method_name

    def plot(self, **kwargs):
        """Plot the attribution result.
        
        Forwards arguments to `sktime_xai.visualization.plot_temporal_heatmap`.
        """
        return plot_temporal_heatmap(
            self.attribution_values, 
            X=self.X_explained, 
            title=f"{self.method_name} Explanation",
            **kwargs
        )

    def __repr__(self):
        return f"AttributionResult(method={self.method_name}, shape={self.attribution_values.shape})"
