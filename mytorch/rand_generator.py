import random


def manual_seed(meta_seed):
    Generator.instance().reset(meta_seed)


class Generator:
    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = Generator()
        return cls._instance

    def __init__(self):
        self.reset(0)

    def reset(self, meta_seed):
        random.seed(meta_seed)

    def generate(self):
        return random.randint(0, int(round(1e9 + 7)))
