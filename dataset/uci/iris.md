# Iris

#UCI

The data set consists of 50 samples from each of three species of Iris (Setosa, Virginica and Versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.

## Load data

### Scikit-learn

```python
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target
```
