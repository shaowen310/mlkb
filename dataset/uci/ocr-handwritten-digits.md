# Optical Recognition of Handwritten Digits Data Set

#UCI

Preprocessing programs made available by NIST is used to extract normalized bitmaps of handwritten digits from a preprinted form. From a total of 43 people, 30 contributed to the training set and different 13 to the test set. 32x32 bitmaps are divided into nonoverlapping blocks of 4x4 and the number of on pixels are counted in each block. This generates an input matrix of 8x8 where each element is an integer in the range 0..16. This reduces dimensionality and gives invariance to small distortions.

|                           |                |
| ------------------------- | -------------- |
| Data Set Characteristics  | Multivariate   |
| Number of Instances       | 5620           |
| Attribute Characteristics | Integer        |
| Number of Attributes      | 64 (8x8)       |
| Date Donated              | 1998-07-01     |
| Associated Tasks          | Classification |
| Missing Values?           | No             |

## Load data

### Scikit-learn

|                   |               |
| ----------------- | ------------- |
| Classes           | 10            |
| Samples per class | ~180          |
| Samples total     | 1797          |
| Dimensionality    | 64            |
| Features          | integers 0-16 |

```python
from sklearn import datasets

digits = datasets.load_digits()

n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
y = digits.target
```

## References

1. [UCI: Optical Recognition of Handwritten Digits Data Set](https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits)
2. [Scikit-learn: sklearn.datasets.load_digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
