from src.imgdataconvertcodegen import is_only_this_key_differ, are_both_same_data_repr


def numpy_rgb_to_bgr(source, target):
    def version_match():
        return True

    def metadata_match(source_metadata, target_metadata):
        if not are_both_same_data_repr(source, target, 'numpy.ndarray'):
            return False

        if not is_only_this_key_differ(source_metadata, target_metadata, 'color_channel'):
            return False

        if (source_metadata.get('color_channel') == 'rgb' and
            target_metadata.get('color_channel') == 'bgr'):
            return True

        return False

    if version_match() and metadata_match(source, target):
        return "def convert(var):\n  return var[:, :, ::-1]"
    return None


def numpy_to_torch(source, target):
    def version_match():
        return True

    def metadata_check(source_metadata, target_metadata):
        if not is_only_this_key_differ(source_metadata, target_metadata, 'data_representation'):
            return False

        return (source.get('data_representation') == 'numpy.ndarray' and
                target.get('data_representation') == 'torch.tensor')

    if version_match() and metadata_check(source, target):
        return "def convert(var):\n  return torch.from_numpy(var)"
    return None


def torch_channel_order_last_to_first(source, target):
    def version_match():
        return True

    def metadata_check(source_metadata, target_metadata):
        if not are_both_same_data_repr(source, target, 'torch.tensor'):
            return False

        if not is_only_this_key_differ(source_metadata, target_metadata, 'channel_order'):
            return False

        return (source_metadata.get('channel_order') == 'channel last' and
                target_metadata.get('channel_order') == 'channel first')

    if version_match() and metadata_check(source, target):
        return "def convert(var):\n  return var.permute(2, 0, 1)"
    return None


def torch_minibatch_input_false_to_true(source, target):
    def version_match():
        return True

    def metadata_check(source_metadata, target_metadata):
        if not are_both_same_data_repr(source, target, 'torch.tensor'):
            return False

        if not is_only_this_key_differ(source_metadata, target_metadata, 'minibatch_input'):
            return False

        return (not source_metadata.get('minibatch_input')) and target_metadata.get('minibatch_input')

    if version_match() and metadata_check(source, target):
        return "def convert(var):\n  return torch.unsqueeze(var, 0)"
    return None


edge_factory_examples = [
    numpy_rgb_to_bgr,
    numpy_to_torch,
    torch_channel_order_last_to_first,
    torch_minibatch_input_false_to_true]


def numpy_bgr_to_rgb(source, target):
    def version_match():
        return True

    def metadata_match(source_metadata, target_metadata):
        if not are_both_same_data_repr(source, target, 'numpy.ndarray'):
            return False

        if not is_only_this_key_differ(source_metadata, target_metadata, 'color_channel'):
            return False

        if (source_metadata.get('color_channel') == 'bgr' and
            target_metadata.get('color_channel') == 'rgb'):
            return True

        return False

    if version_match() and metadata_match(source, target):
        return "def convert(var):\n  return var[:, :, ::-1]"
    return None


new_edge_factory = numpy_bgr_to_rgb
