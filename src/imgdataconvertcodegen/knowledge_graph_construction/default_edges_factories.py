from ..metadata_differ import are_both_same_data_repr

# NOTE: the source and target metadata are only different in one attribute


def numpy_between_rgb_bgr(source_metadata, target_metadata):
    if not are_both_same_data_repr(source_metadata, target_metadata, 'numpy.ndarray'):
        return None
    if ((source_metadata.get('color_channel') == 'bgr' and
        target_metadata.get('color_channel') == 'rgb') or
        (source_metadata.get('color_channel') == 'rgb' and
         target_metadata.get('color_channel') == 'bgr')):
        return "def convert(var):\n  return var[:, :, ::-1]"
    return None


def numpy_to_torch(source_metadata, target_metadata):
    if (source_metadata.get('data_representation') == 'numpy.ndarray' and
            target_metadata.get('data_representation') == 'torch.tensor'):
        return "def convert(var):\n  return torch.from_numpy(var)"
    return None


def torch_channel_order_last_to_first(source_metadata, target_metadata):
    if not are_both_same_data_repr(source_metadata, target_metadata, 'torch.tensor'):
        return None
    if (source_metadata.get('channel_order') == 'channel last' and
            target_metadata.get('channel_order') == 'channel first'):
        if source_metadata.get('minibatch_input'):
            return "def convert(var):\n  return var.permute(0, 3, 1, 2)"
        return "def convert(var):\n  return var.permute(2, 0, 1)"
    return None


def torch_minibatch_input_false_to_true(source_metadata, target_metadata):
    if not are_both_same_data_repr(source_metadata, target_metadata, 'torch.tensor'):
        return None
    if (not source_metadata.get('minibatch_input')) and target_metadata.get('minibatch_input'):
        return "def convert(var):\n  return torch.unsqueeze(var, 0)"

    return None


# Todo... add more factories from the table
default_factories = [
    numpy_between_rgb_bgr,
    numpy_to_torch,
    torch_channel_order_last_to_first,
    torch_minibatch_input_false_to_true]
