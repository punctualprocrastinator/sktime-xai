
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
print(f"Test Score: {clf.score(X_test, y_test)}")

if hasattr(clf, "univar_rocket_"):
    ur = clf.univar_rocket_
    print(f"Main Script: univar_rocket_ found. Type: {type(ur)}")
    if hasattr(ur, "classifier"):
        c = ur.classifier
        print(f"Main Script: classifier found. Type: {type(c)}")
        print(f"Main Script: classifier dir: {dir(c)}")
        if hasattr(c, "coef_"):
             print(f"Main Script: coef_ shape: {c.coef_.shape}")
        else:
             print("Main Script: classifier has NO coef_")

# Select a test instance
idx = 0
X_instance = X_test.iloc[[idx]]
true_class = y_test[idx]
print(f"Instance {idx}, True Class: {true_class}")

# DEBUG: Inspect estimator before explainer
print("DEBUG: Inspecting clf.estimator_ in verify_rocket.py")
est = clf.estimator_
steps = getattr(est, "steps", getattr(est, "_steps", None))
with open("verify_debug.log", "w") as f:
    for name, step in steps:
        f.write(f"Step: {name}, Type: {type(step)}\n")
        f.write(f"Dirs: {[d for d in dir(step) if not d.startswith('__')]}\n")
        if hasattr(step, "coef_"):
            f.write(f"  -> Has coef_! Shape: {step.coef_.shape}\n")
        else:
            f.write(f"  -> NO coef_!\n")
            f.write(f"  -> Dict keys: {list(step.__dict__.keys())}\n")
            try:
                f.write(f"  -> Trying to access coef_: {step.coef_}\n")
            except Exception as e:
                f.write(f"  -> Accessing coef_ failed: {e}\n")

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
