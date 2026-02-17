"""Base class for all sktime-xai explainers."""

from sktime.base import BaseObject


class BaseExplainer(BaseObject):
    """Base class for explainers."""

    _tags = {
        "object_type": "explainer",
        "authors": ["sktime-xai-developers"],
        "maintainers": ["sktime-xai-developers"],
        "task": "forecasting",  # forecasting, classification, etc.
        "modality": "time-series",
    }

    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        """Fit the explainer.
        
        Parameters
        ----------
        X : pd.DataFrame or pd.Series
            Background data or training data used to initialize the explainer.
        y : pd.Series, optional
            Target data, by default None
            
        Returns
        -------
        self : object
            Reference to self.
        """
        raise NotImplementedError("Abstract method 'fit' not implemented")

    def explain(self, X, y=None, **kwargs):
        """Generate explanations for X.

        Parameters
        ----------
        X : pd.DataFrame or pd.Series
            New data to explain.
        y : pd.Series, optional
            Target data useful for some explainers (e.g. alignment), by default None

        Returns
        -------
        AttributionResult
            The attribution result object containing scores and plotting methods.
        """
        raise NotImplementedError("Abstract method 'explain' not implemented")
