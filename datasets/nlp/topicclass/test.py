# %%
import itertools

from agnews import AGNewsData

agnewsdata = AGNewsData()

d_train = agnewsdata.generate_samples(agnewsdata.train_file)

for r in itertools.islice(d_train, 5):
    print(r)
# %%