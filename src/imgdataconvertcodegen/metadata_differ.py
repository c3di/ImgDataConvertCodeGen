def are_both_same_data_repr(metadata_a, metadata_b, data_repr):
    return metadata_a.get('data_representation') == data_repr and metadata_b.get('data_representation') == data_repr


def is_single_metadata_differ(metadata_a, metadata_b, not_included_keys=None):
    metadata_a = metadata_a.copy()
    metadata_b = metadata_b.copy()
    if not_included_keys:
        for key in not_included_keys:
            metadata_a.pop(key, None)
            metadata_b.pop(key, None)

    if set(metadata_a) != set(metadata_b):
        return False

    differences = 0
    for key in metadata_a:
        if metadata_a[key] != metadata_b[key]:
            differences += 1
            if differences > 1:
                return False

    return differences == 1


def is_same_metadata(metadata_a, metadata_b, not_included_keys=None):
    metadata_a = metadata_a.copy()
    metadata_b = metadata_b.copy()
    if not_included_keys:
        for key in not_included_keys:
            metadata_a.pop(key, None)
            metadata_b.pop(key, None)

    return metadata_a == metadata_b


def is_differ_value_for_key(metadata_a, metadata_b, key):
    return metadata_a[key] != metadata_b[key]
