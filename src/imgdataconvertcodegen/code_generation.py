import uuid

from .util import extract_func_body
from .knowledge_graph_construction import encode_metadata


class ConvertCodeGenerator:
    _knowledge_graph = None
    _cache = {}

    def __init__(self, knowledge_graph):
        self._knowledge_graph = knowledge_graph
        self._cache = {}

    @property
    def knowledge_graph(self):
        return self._knowledge_graph

    @knowledge_graph.setter
    def knowledge_graph(self, value):
        self._knowledge_graph = value

    def get_convert_path(self, source_spec: str | dict, target_spec: str | dict,
                         source_color_channel='color', target_color_channel='color'):
        path = self.knowledge_graph.get_shortest_path(self._get_metadata(source_spec, source_color_channel),
                                                      self._get_metadata(target_spec, target_color_channel))
        if path is None:
            return []
        metadata_list = []
        for node_id in path:
            metadata_list.append(self.knowledge_graph.get_node(node_id))
        return metadata_list

    def get_conversion(self, source_var_name: str, source_spec, target_var_name: str, target_spec,
                       source_color_channel='color', target_color_channel='color') -> str | None:
        """
        Generates Python code as a string that performs data conversion from a source variable to a target variable
        Args:
            source_var_name: the name of the variable holding the source data.
            source_spec: the name of library or a dictionary containing metadata about the source data.
            target_var_name:  the name of the variable that will store the result of the conversion.
            target_spec: the same as source_spec
            source_color_channel: the color channel of the source data if the source_spec is a library name.
                the value could be 'gray' | 'color' | None
            target_color_channel: the same as source_color_channel

        Returns: A string containing the Python code necessary to perform the conversion.

        """
        return self.get_conversion_using_metadata(source_var_name,
                                                  self._get_metadata(source_spec, source_color_channel),
                                                  target_var_name,
                                                  self._get_metadata(target_spec, target_color_channel))

    def get_conversion_using_metadata(self, source_var_name, source_metadata,
                                      target_var_name: str, target_metadata) -> str | None:
        """
        Generates Python code as a string that performs data conversion from a source variable to a target variable
         based on the provided metadata.

        Parameters:
            source_var_name (str): The name of the variable holding the source data.
            source_metadata (dict): A dictionary containing metadata about the source data, such as color channels, etc.
            target_var_name (str): The name of the variable that will store the result of the conversion.
            target_metadata (dict): A dictionary containing metadata about the target data.

        Examples:
            >>> source_var_name = "source_image"
            >>> source_metadata = {"color_channel": "bgr", "channel_order": "channel last", ...}
            >>> target_var_name = "target_image"
            >>> target_metadata = {"color_channel": "rgb", "channel_order": "channel first", ...}
            >>> convert_code_generator = ConvertCodeGenerator()
            >>> conversion = convert_code_generator.get_conversion_using_metadata(source_var_name, source_metadata,
            >>> target_var_name, target_metadata)
            >>> conversion
            ('', '# Convert BGR to RGB\nvar1 = source_image[:, :, ::-1]\n# Change data format from HWC to CHW\nvar2 = np.transpose(var1, (2, 0, 1))\ntarget_image = var2')
        """
        source_encode_str = encode_metadata(source_metadata)
        target_encode_str = encode_metadata(target_metadata)
        if (source_encode_str, target_encode_str) in self._cache:
            return self._cache[(source_encode_str, target_encode_str)]

        cvt_path = self.knowledge_graph.get_shortest_path(source_metadata, target_metadata)
        if cvt_path is None:
            result = None
        elif len(cvt_path) == 1:
            result = f"{target_var_name} = {source_var_name}"
        else:
            result = self._get_conversion_multiple_steps(cvt_path, source_var_name, target_var_name)
        self._cache[(source_encode_str, target_encode_str)] = result
        return result

    def _get_conversion_multiple_steps(self, cvt_path_in_kg, source_var_name, target_var_name) -> str:
        imports = set()
        main_body = []
        arg = source_var_name
        for i in range(len(cvt_path_in_kg) - 1):
            return_name = f"var_{uuid.uuid4().hex}" if i != len(cvt_path_in_kg) - 2 else target_var_name
            imports_step, main_body_step = self._get_conversion_per_step(cvt_path_in_kg[i], cvt_path_in_kg[i + 1],
                                                                         arg, return_name)
            if imports_step != '':
                imports.update(imports_step.split('\n'))
            main_body.append(main_body_step)
            arg = return_name
        return '\n'.join(main_body) if len(imports) == 0 else '\n'.join(imports) + '\n' + '\n'.join(main_body)

    def _get_conversion_per_step(self, source_id, target_id, arg, return_name):
        conversion_on_edge = self.knowledge_graph.get_edge_data(source_id, target_id)['conversion']
        imports = conversion_on_edge[0]
        main_body = extract_func_body(conversion_on_edge[1], arg, return_name)
        return imports, main_body

    def _get_metadata(self, spec, color_channel='color'):
        if isinstance(spec, dict):
            return spec
        return self.knowledge_graph.get_metadata_by_lib_name(spec, color_channel)
