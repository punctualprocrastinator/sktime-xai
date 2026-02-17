
import numpy as np
import pandas as pd
from sktime.classification.kernel_based import RocketClassifier

print("Creating dummy classifier...")
clf = RocketClassifier(num_kernels=10)
X = pd.DataFrame(pd.Series([pd.Series(np.random.randn(10))]))
y = pd.Series([0])
clf.fit(X, y)

est = clf.estimator_
if hasattr(est, "_steps"):
    print(f"_steps found: {type(est._steps)}")
    for name, step in est._steps:
        print(f"Step '{name}': {type(step)}")
        if hasattr(step, "kernels"):
            print("  -> Has kernels!")
        if hasattr(step, "coef_"):
            print("  -> Has coef_!")
else:
    print("_steps NOT found.")
