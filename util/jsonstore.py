import collections
import os
import json
import uuid


class KeyGen:
    def __iter__(self):
        return self

    def __next__(self):
        return str(uuid.uuid4())[:8]


class JsonStore(collections.MutableMapping):
    def __init__(self, root_dir):
        super().__init__()
        self.keygen = KeyGen()
        self.root_dir = root_dir

    def __getitem__(self, key):
        fp = os.path.join(self.root_dir, key + '.json')
        with open(fp, "r") as file:
            return json.load(file)

    def __setitem__(self, key):
        pass

    def __delitem__(self, key):
        pass

    def __iter__(self):
        pass

    def __len__(self):
        pass

    def keys(self):
        keyset = set()
        for _, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.json'):
                    keyset.add(os.path.splitext(file)[0])
        return keyset
