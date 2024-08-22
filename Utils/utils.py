
def ensureList(obj):
    if isinstance(obj, (tuple, list)):
        return obj
    return [obj]
