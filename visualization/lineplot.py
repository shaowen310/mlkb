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
h = 0.01
x_a = np.arange(x.min(), x.max(), h)
y_pred = linearreg.predict(x_a.reshape(-1, 1)).ravel()

# %%
fig, axes = plt.subplots(1, 1)

x_pos = np.arange(0, 4, 1)
y = np.array([0.796, 0.808, 0.809, 0.805])

# line
axes.plot(x_pos, y)
plt.xticks(x_pos, ['50', '100', '150', '200'])
# dots
# axes.scatter(x[:, 0], y[:, 0])

plt.show()

# %%
