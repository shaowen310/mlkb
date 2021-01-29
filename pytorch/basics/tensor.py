# %%
import torch

# %%
# Tensor initialiation
# by instance
a0 = torch.tensor(1, dtype=torch.float)
a1 = torch.tensor((1,1)).float()
# by size
b0 = torch.zeros((1,))
# range
c0 = torch.arange(0, 3)
# random
r0 = torch.rand((3,))
r1 = torch.randint(0, 3, (3,))
r2 = torch.randperm(3)

# %%
# Transform
x = torch.ones((1,2,3))
# reshape
a0 = x.view((1,-1))
# transpose
t0 = x.T # size ((3,2,1))
t1 = torch.transpose(x, 0, 1)
# %%
# Aggregation
x = torch.arange(0,9).view((3,3))
mi0 = x.min()
mi1 = x.min(dim=0)
mi2 = torch.min(x, dim=1)
# min, max, mean

# %%
# Math
x = torch.arange(0,6).view((2,3))
y = torch.arange(0,6).view((3,2))
ele_prod = x * y.T
matmul = x @ y
matmul2 = torch.matmul(x, y)
# %%
