def flatten(d: dict) -> dict:
    """
    Nested dict to flat dict. E.g.
    {key1: {key2: value}} -> {key1.key2: value}
    """
    flattened_d = dict()
    for key, value in d.items():
        if isinstance(value, dict):
            for nested_key, flattened_values in flatten(value).items():
                flattened_d[f"{key}.{nested_key}"] = flattened_values
        else:
            flattened_d[key] = value
    return flattened_d


def unflatten(flattened_d: dict) -> dict:
    """
    Flat dict to nested dict. E.g.
    {key1.key2: value} -> {key1: {key2: value}}
    """
    unflattened_d = dict()
    for key, value in flattened_d.items():
        parts = key.split('.')
        d = unflattened_d
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return unflattened_d

def merge_dicts(dict_1: dict, dict_2: dict):
    """
    Merge two dicts. E.g.
    {key1: {key2: value}} + {key1.key3: value} -> {key1: {key2: value, key3: value}}
    """
    flat_merged_dict = {**flatten(dict_1), **flatten(dict_2)}
    return unflatten(flat_merged_dict)
