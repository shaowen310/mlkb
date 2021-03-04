# %%
import itertools

from agnews import AGNewsData

agnews = AGNewsData()

d_train = agnews.generate_samples(agnews.train_file)

for r in itertools.islice(d_train, 5):
    print(r)
# %%