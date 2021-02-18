# %%
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# %%
# Data creation
cancer = datasets.load_breast_cancer()
x = cancer['data'][:100]
y = cancer['target'][:100]

stdscale = StandardScaler()
pca = PCA(n_components=2)
x_pca = pca.fit_transform(stdscale.fit_transform(x))

svmclf = SVC(kernel='rbf')
svmclf.fit(x_pca, y)

# %%
# Decision boundary
fig, axes = plt.subplots(1, 1)

h = 0.01
x_min, x_max = x_pca[:, 0].min() - .1, x_pca[:, 0].max() + .1
y_min, y_max = x_pca[:, 1].min() - .1, x_pca[:, 1].max() + .1
# point grids
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# concatenate
x_mesh = np.c_[xx.ravel(), yy.ravel()]
z = svmclf.predict(x_mesh)
z = z.reshape(xx.shape)
axes.set_title('Decision Boundary', fontsize=20)
axes.contourf(xx, yy, z)
# axes.axis('off')

# Data points
N = 100
CLS_NUM = 2
markers = ['o', 'x']
for i in range(CLS_NUM):
    x_cls_i = x_pca[y == i]
    axes.scatter(x_cls_i[:, 0], x_cls_i[:, 1], s=40, marker=markers[i])

plt.show()

# %%
