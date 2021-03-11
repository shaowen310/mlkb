import uuid


class UUIDv4Gen:
    def __init__(self, usefirstnchars=8):
        self.usefirstnchars = usefirstnchars

    def __iter__(self):
        return self

    def __next__(self):
        id = str(uuid.uuid4())
        return id[:self.usefirstnchars] if self.usefirstnchars > 0 else id


class IntIDGen:
    def __init__(self, start=0):
        self.start = start

    def __iter__(self):
        self.id = self.start - 1
        return self

    def __next__(self):
        self.id += 1
        return self.id
