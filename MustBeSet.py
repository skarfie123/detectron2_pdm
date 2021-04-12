class MustBeSet(type):
    def __getattribute__(cls, key):
        if super().__getattribute__(key) is None:
            raise Exception("Config has not been set, use CustomConfig.set()")
        return super().__getattribute__(key)
