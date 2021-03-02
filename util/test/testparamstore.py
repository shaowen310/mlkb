# %%
from paramstore import ParamStore

# %%
store = ParamStore('./param_')

keys = store.keys()

test_json = store['testfile']

print(test_json)

# %%
test_model_obj = {'model': 1}
store.add('test_model', test_model_obj)

# %%
