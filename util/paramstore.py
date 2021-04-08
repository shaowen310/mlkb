import collections
import os
import json
import uuid


class UUIDv4Gen:
    def __iter__(self):
        return self

    def __next__(self):
        return str(uuid.uuid4())[:8]


class JsonStore(collections.MutableMapping):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir

        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

    def _getfp(self, key):
        return os.path.join(self.root_dir, key + '.json')

    def __getitem__(self, key):
        try:
            obj = None
            with open(self._getfp(key), "r") as file:
                obj = json.load(file)
        except FileNotFoundError:
            pass

        return obj

    def __setitem__(self, key, obj):
        with open(self._getfp(key), 'w') as file:
            json.dump(obj, file)

    def __delitem__(self, key):
        fp = self._getfp(key)
        if os.path.exists(fp):
            os.remove(fp)

    def __iter__(self):
        raise NotImplementedError()

    def __len__(self):
        return len(self.keys())

    def __contains__(self, key):
        return os.path.exists(self._getfp(key))

    def keys(self):
        keyset = set()
        for _, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.json'):
                    keyset.add(os.path.splitext(file)[0])
        return keyset

    def items(self):
        raise AttributeError()

    def values(self):
        raise AttributeError()

    def get(self, key):
        return self[key]


class ParamStore(JsonStore):
    def __init__(self, root_dir='param_'):
        super().__init__(root_dir)
        self.idgen = UUIDv4Gen()

    def add(self, model_name, obj):
        for key in map(lambda randkey: '_'.join((model_name, randkey)), self.idgen):
            if key not in self:
                obj['id_'] = key
                self[key] = obj
                return key
