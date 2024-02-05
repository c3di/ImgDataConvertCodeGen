def factory(source, target):
    def version_match():
        return True

    def metadata_match(source_metadata, target_metadata):
        if source.get('data_representation') != 'numpy.ndarray':
            return False

        source_metadata_copy = source_metadata.copy()
        target_metadata_copy = target_metadata.copy()
        source_metadata_copy.pop('color_channel', None)
        target_metadata_copy.pop('color_channel', None)

        if source_metadata_copy != target_metadata_copy:
            return False

        if (source_metadata.get('color_channel') == 'BGR' and
            target_metadata.get('color_channel') == 'RGB') or \
           (source_metadata.get('color_channel') == 'RGB' and
                target_metadata.get('color_channel') == 'BGR'):
            return True

        return False

    if version_match() and metadata_match(source, target):
        return "def convert(value):\n  return value[:, :, ::-1]"
    return None

def example_factory2(source, target):
    return "def convert(value):\n  return value[:, :, ::-1]"


factories = [
    factory,
    example_factory2]
