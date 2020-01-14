# IMDB sentiment analysis

## Load data

Modify `imdb.py` in keras library

```
-  with np.load(path) as f:
+  with np.load(path, allow_pickle=True) as f:
```

```
(X_train, y_train), (X_test, y_test) = imdb.load_data()
```

X_train, X_test 25,000 encoded word sequence lists.
