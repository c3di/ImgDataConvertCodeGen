from .type import ImgRepr, PossibleValuesForImgRepr, ValidCheckFunc, img_metadata_config, Metadata


def add_img_metadata_config(img_repr: ImgRepr, possible_values: PossibleValuesForImgRepr,
                            valid_check: ValidCheckFunc):
    img_metadata_config[img_repr] = (possible_values, valid_check)


def is_valid_attribute_value(value: Metadata, valid_values: PossibleValuesForImgRepr):
    for attribute, values in valid_values.items():
        if value[attribute] not in values:
            return False
    return True


def find_closest_metadata(source_metadata, candidates):
    if len(candidates) == 0:
        return None
    if len(candidates) == 1:
        return candidates[0]

    targets = candidates
    targets = [
        candidate for candidate in targets
        if candidate["data_representation"] == source_metadata["data_representation"]
    ]
    if len(targets) == 0:
        targets = candidates

    color_matched_targets = [
        candidate for candidate in targets
        if candidate["color_channel"] == source_metadata["color_channel"]
    ]
    if len(color_matched_targets) == 0:
        if source_metadata["color_channel"] in ["rgb", "bgr"]:
            for metadata in targets:
                if metadata["color_channel"] in ["rgb", "bgr"]:
                    return metadata
    return targets[0]


def encode_metadata(metadata: dict) -> str:
    return '-'.join([str(metadata[key]) for key in Metadata.__annotations__.keys()])


def decode_metadata(metadata_str: str) -> dict:
    metadata_list = metadata_str.split('-')
    metadata = {k: v for k, v in zip(list(Metadata.__annotations__.keys()), metadata_list)}
    metadata['minibatch_input'] = True if metadata['minibatch_input'] == 'True' else False
    return metadata
