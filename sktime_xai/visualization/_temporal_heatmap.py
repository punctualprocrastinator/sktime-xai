"""Visualization utilities for time series attribution."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_temporal_heatmap(
    attribution_values, 
    X=None, 
    ax=None, 
    title="Temporal Attribution",
    cmap="coolwarm",
    alpha=0.8
):
    """Plot attribution values as a heatmap or colored line.
    
    Parameters
    ----------
    attribution_values : pd.Series, pd.DataFrame, or np.ndarray
        The attribution scores for each timestep/feature.
        Shape should be (n_timepoints,) or (n_variables, n_timepoints).
    X : pd.Series or pd.DataFrame, optional
        The original time series data. If provided, it will be plotted 
        overlaying the attribution (or above it).
    ax : matplotlib.axes.Axes, optional
        Target axes, by default None.
    
    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    # Standardization to DataFrame
    if isinstance(attribution_values, pd.Series):
        vals = attribution_values.values.reshape(1, -1)
        columns = attribution_values.index
    elif isinstance(attribution_values, pd.DataFrame):
        vals = attribution_values.values.T  # (n_vars, n_time)
        columns = attribution_values.index
    else:
        vals = np.array(attribution_values)
        if vals.ndim == 1:
            vals = vals.reshape(1, -1)
    
    # Normalize for color (symmetric around 0)
    limit = np.nanmax(np.abs(vals))
    
    # Plot heatmap
    im = ax.imshow(
        vals, 
        aspect='auto', 
        cmap=cmap, 
        vmin=-limit, 
        vmax=limit, 
        alpha=alpha,
        extent=[0, vals.shape[1], 0, vals.shape[0]]
    )
    
    # Overlay original signal if provided (simple scaling for viz)
    if X is not None:
        # Create twin axis for signal
        ax2 = ax.twinx()
        if isinstance(X, (pd.Series, pd.DataFrame)):
            ax2.plot(X.values, color='black', linewidth=1.5, label='Signal')
        else:
            ax2.plot(X, color='black', linewidth=1.5, label='Signal')
        ax2.set_ylabel("Signal Value")
        # ax2.legend(loc='upper right')

    ax.set_title(title)
    ax.set_xlabel("Time Step")
    ax.set_yticks([])  # Hide y ticks for now as it's abstract
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical')
    cbar.set_label("Attribution Score")
    
    return fig
