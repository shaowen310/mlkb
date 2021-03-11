# %%
import torch
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# %%
# Data
x = torch.randn(100, 1)
y = 5 * x + torch.randn(100, 1)
linearreg = LinearRegression()
linearreg.fit(x, y)

# %%
h = 0.01

fig, axes = plt.subplots(1, 1)
x_a = np.arange(x.min(), x.max(), h)
# line
axes.plot(x_a, linearreg.predict(x_a.reshape(-1, 1)).ravel())
# dots
axes.scatter(x[:, 0], y[:, 0])

plt.show()

# %%
