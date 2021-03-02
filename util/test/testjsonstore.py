# %%
from jsonstore import JsonStore

# %%
json_store = JsonStore('./param_')

keys = json_store.keys()

test_json = json_store['testfile']

print(test_json)

# %%