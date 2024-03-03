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


def is_only_this_key_differ(metadata_a, metadata_b, key):
    if metadata_a[key] != metadata_b[key]:
        return is_same_metadata(metadata_a, metadata_b, [key])
    return False


# !!! Do not use it! This is only for is_data_type_and_intensity_range_differ()
def __is_two_keys_differ(metadata_a, metadata_b, first_key, second_key):
    if metadata_a[first_key] != metadata_b[first_key] and metadata_a[second_key] != metadata_b[second_key]:
        return is_same_metadata(metadata_a, metadata_b, [first_key, second_key])
    return False


# !!! Only use this for floats
def is_data_type_and_intensity_range_differ(metadata_a, metadata_b):
    __is_two_keys_differ(metadata_a, metadata_b, "data_type", "intensity_range")
