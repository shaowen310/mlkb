# %%
import torch
import torch.nn as nn

# %%
x = torch.rand((10, 10))
w = torch.rand((10, 10), requires_grad=True)
b = torch.rand((10, ), requires_grad=True)

# Loss
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.SGD((w, b), lr=0.01)

last_layer = w @ x + b
labels = torch.randint(0, 10, (10, ))

loss = criterion(last_layer, labels)

# Sets the gradients of w, b to 0
optimizer.zero_grad()

# Calculate the gradients
loss.backward()

# dw/dx
w_grad = w.grad
# db/dx
b_grad = b.grad

# One-step update of w, b using the gradients
optimizer.step()

# %%
