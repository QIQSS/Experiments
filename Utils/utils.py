
def ensureList(obj):
    if isinstance(obj, (tuple, list)):
        return obj
    return [obj]
    
def mergeDict(dict1, dict2):
    """Merge dict2 into dict1, with dict2 values overriding dict1 values."""
    result = dict1.copy()
    result.update(dict2)
    return result


class customDict(dict):
    """ custom version of a dictionnary
    
    new method:
        rget: recursive get, return first val with key in child dicts
        
    """
    
    def __init__(self, *arg, **kw):
        super(customDict, self).__init__(*arg, **kw)

    def rget(self, key, default=None):
        """ recursive get:
            search inside values that are dict.
            return the first one find
            or default
        """    
        def _searchKey(dic, key_to_find):
            for key, val in dic.items():
                if isinstance(val, (dict, customDict)):
                    return _searchKey(val, key_to_find)
                if key == key_to_find:
                    return val
            return default
        return _searchKey(self, key)
