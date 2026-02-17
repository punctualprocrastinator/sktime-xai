
import numpy as np
import pandas as pd
from sktime.transformations.panel.rocket import Rocket

# Generate dummy data
X = pd.DataFrame(pd.Series([pd.Series(np.random.randn(100))]))
y = pd.Series([0])

print("Initializing Rocket...")
rocket = Rocket(num_kernels=100, random_state=42)
print("Fitting Rocket...")
rocket.fit(X)

print("Rocket attributes:")
print(dir(rocket))

if hasattr(rocket, "kernels"):
    print("Found 'kernels' attribute!")
    # inspect kernel structure
    # usually (weights, length, bias, dilation, padding)
    print(rocket.kernels)
else:
    print("'kernels' attribute NOT found.")
    # Check for underlying minirocket or similar
