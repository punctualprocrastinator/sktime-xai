
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sktime.datasets import load_gunpoint
from sktime.classification.kernel_based import RocketClassifier
from sktime_xai.classification._rocket_explainer import RocketExplainer

print("Loading GunPoint data...")
X_train, y_train = load_gunpoint(split="train")
X_test, y_test = load_gunpoint(split="test")

# Using a small number of kernels for speed
print("Training RocketClassifier...")
clf = RocketClassifier(num_kernels=100, random_state=42)
clf.fit(X_train, y_train)
print("Training complete.")

# Select a test instance
idx = 0
X_instance = X_test.iloc[[idx]]
true_class = y_test[idx]
print(f"Instance {idx}, True Class: {true_class}")

print("Initializing RocketExplainer...")
explainer = RocketExplainer(clf)

# In GunPoint, classes are '1' and '2'. 
# RocketClassifier maps them to internal indices.
# We explain the first class index (0) corresponding to one of the classes.
pred_class_idx = 0 

print(f"Explaining class index {pred_class_idx}...")
explanation = explainer.explain(X_instance, class_index=pred_class_idx, top_k=3)

print("Explanation generated.")
print("Top kernels info:")
for i, info in enumerate(explanation.top_kernels_info):
    print(f"Rank {i+1}: Kernel {info['kernel_idx']} ({info['type']}) | Contribution: {info['contribution']:.4f}")

# Plot (headless)
fig = explanation.plot()
print("Plot created successfully.")
print("ROCKET XAI verified!")
