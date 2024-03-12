from imgdataconvertcodegen import are_both_same_data_repr, is_single_metadata_differ, is_same_metadata, \
    is_differ_value_for_key


def test_both_metadata_match_data_repr():
    metadata_a = {'data_representation': 'torch.tensor'}
    metadata_b = {'data_representation': 'torch.tensor'}
    data_repr = 'torch.tensor'
    assert are_both_same_data_repr(metadata_a, metadata_b, data_repr), \
        "Should return True when both metadata have the same data_representation matching data_repr"


def test_one_metadata_missing_data_repr():
    metadata_a = {'data_representation': 'torch.tensor'}
    metadata_b = {}
    data_repr = 'torch.tensor'
    assert not are_both_same_data_repr(metadata_a, metadata_b, data_repr), \
        "Should return False when one metadata is missing the data_representation key"


def test_one_metadata_different_data_repr():
    metadata_a = {'data_representation': 'torch.tensor'}
    metadata_b = {'data_representation': 'numpy.ndarray'}
    data_repr = 'torch.tensor'
    assert not are_both_same_data_repr(metadata_a, metadata_b, data_repr), \
        "Should return False when one metadata has a different data_representation value"


def test_both_metadata_missing_data_repr():
    metadata_a = {}
    metadata_b = {}
    data_repr = 'torch.tensor'
    assert not are_both_same_data_repr(metadata_a, metadata_b, data_repr), \
        "Should return False when both metadata are missing the data_representation key"


def test_both_metadata_different_data_repr():
    metadata_a = {'data_representation': 'torch.tensor'}
    metadata_b = {'data_representation': 'numpy.ndarray'}
    data_repr = 'PIL'
    assert not are_both_same_data_repr(metadata_a, metadata_b, data_repr), \
        "Should return False when both metadata have different data_representation values not matching data_repr"


def test_single_difference():
    metadata_a = {'key1': 'value1', 'key2': 'value2'}
    metadata_b = {'key1': 'value1', 'key2': 'different_value'}
    assert is_single_metadata_differ(metadata_a, metadata_b), \
        "Should return True when there is exactly one differing key-value pair"


def test_no_differences():
    metadata_a = {'key1': 'value1', 'key2': 'value2'}
    metadata_b = {'key1': 'value1', 'key2': 'value2'}
    assert not is_single_metadata_differ(metadata_a, metadata_b), \
        "Should return False when there are no differing key-value pairs"


def test_more_than_one_difference():
    metadata_a = {'key1': 'value1', 'key2': 'value2'}
    metadata_b = {'key1': 'different_value1', 'key2': 'different_value2'}
    assert not is_single_metadata_differ(metadata_a, metadata_b), \
        "Should return False when there are more than one differing key-value pairs"


def test_key_absent_in_one():
    metadata_a = {'key1': 'value1', 'key2': 'value2'}
    metadata_b = {'key1': 'value1', 'key3': 'value3'}
    assert not is_single_metadata_differ(metadata_a, metadata_b), \
        "Should return False when exactly one key is absent in one of the dictionaries"


def test_multiple_keys_absent():
    metadata_a = {'key1': 'value1', 'key2': 'value2'}
    metadata_b = {'key1': 'value1', 'key3': 'value3', 'key4': 'value4'}
    assert not is_single_metadata_differ(metadata_a, metadata_b), \
        "Should return False when more than one key is absent in one of the dictionaries"


def test_empty_dictionaries():
    metadata_a = {}
    metadata_b = {}
    assert not is_single_metadata_differ(metadata_a, metadata_b), \
        "Should return False when both metadata dictionaries are empty"


def test_one_empty_dictionary():
    metadata_a = {'key1': 'value1'}
    metadata_b = {}
    assert not is_single_metadata_differ(metadata_a, metadata_b), \
        "Should return False when one metadata dictionary is empty and the other is not"


def test_is_single_metadata_differ_with_excluded_keys():
    metadata_a = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}
    metadata_b = {'key1': 'value1', 'key2': 'diff_value2', 'key3': 'diff_value3'}
    not_included_keys = ['key2']

    assert is_single_metadata_differ(metadata_a, metadata_b, not_included_keys), \
        "Should return True when excluding the differing key"


def test_is_single_metadata_differ_with_multiple_excluded_keys():
    metadata_a = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3', 'key4': 'value4'}
    metadata_b = {'key1': 'value1', 'key2': 'diff_value2', 'key3': 'diff_value3', 'key4': 'diff_value4'}
    not_included_keys = ['key2', 'key3']

    assert is_single_metadata_differ(metadata_a, metadata_b, not_included_keys), \
        "Should return True when excluding keys with differing values"


def test_is_single_metadata_differ_with_excluded_nonexistent_keys():
    metadata_a = {'key1': 'value1', 'key2': 'value2'}
    metadata_b = {'key1': 'diff_value1', 'key2': 'value2'}
    not_included_keys = ['key3']

    assert is_single_metadata_differ(metadata_a, metadata_b, not_included_keys), \
        ("Should return True: There's exactly one differing key, and excluding a nonexistent key should not affect the "
         "outcome.")


def test_is_same_metadata_identical():
    metadata_a = {'key1': 'value1', 'key2': 'value2'}
    metadata_b = {'key1': 'value1', 'key2': 'value2'}
    assert is_same_metadata(metadata_a, metadata_b), "Metadata dictionaries should be considered identical"


def test_is_same_metadata_with_excluded_keys():
    metadata_a = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}
    metadata_b = {'key1': 'value1', 'key2': 'diff_value2', 'key3': 'value3'}
    assert is_same_metadata(metadata_a, metadata_b, not_included_keys=['key2']), \
        "Metadata dictionaries should be considered identical when excluding specified keys"


def test_is_same_metadata_different():
    metadata_a = {'key1': 'value1', 'key2': 'value2'}
    metadata_b = {'key1': 'value1', 'key2': 'diff_value2'}
    assert not is_same_metadata(metadata_a, metadata_b), \
        "Metadata dictionaries with differences not in excluded keys should not be considered identical"


def test_is_differ_value_for_key_true():
    metadata_a = {'key1': 'value1', 'key2': 'value2'}
    metadata_b = {'key1': 'diff_value1', 'key2': 'value2'}
    assert is_differ_value_for_key(metadata_a, metadata_b, 'key1'), \
        "Should return True when only the specified key differs"


def test_is_differ_value_for_key_false():
    metadata_a = {'key1': 'value1', 'key2': 'value2'}
    metadata_b = {'key1': 'value1', 'key2': 'value2'}
    assert not is_differ_value_for_key(metadata_a, metadata_b, 'key1'), \
        "Should return False when there are no differences, even for the specified key"
