# %%
import torch

# %%
# Autograd
x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

y = w * x + b

y.backward()

x_grad = x.grad
w_grad = w.grad
b_grad = b.grad
