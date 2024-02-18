import uuid

from src.imgdataconvertcodegen.function_util import extract_func_body
from src.imgdataconvertcodegen.knowledge_graph_construction import encode_to_string, conversion


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

    def get_convert_path(self, source_spec: str | dict, target_spec: str | dict):
        path = self.knowledge_graph.get_shortest_path(self._get_metadata(source_spec), self._get_metadata(target_spec))
        metadata_list = []
        for node_id in path:
            metadata_list.append(self.knowledge_graph.get_node(node_id))
        return metadata_list

    def get_conversions(self, source_metadata, target_metadata) -> list[conversion] | None:
        source_encode_str = encode_to_string(source_metadata)
        target_encode_str = encode_to_string(target_metadata)
        if (source_encode_str, target_encode_str) in self._cache:
            return self._cache[(source_encode_str, target_encode_str)]
        path = self.knowledge_graph.get_shortest_path(source_metadata, target_metadata)
        if path is None:
            return None
        conversions = []
        for i in range(len(path) - 1):
            edge = self.knowledge_graph.get_edge(path[i], path[i + 1])
            conversions.append(edge['conversion'])
        self._cache[(source_encode_str, target_encode_str)] = conversions
        return conversions

    def generate_conversion_using_metadata(self, source_var_name, source_metadata,
                                           target_var_name: str, target_metadata) -> conversion:
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
            >>> conversion = convert_code_generator.generate_conversion_using_metadata(source_var_name, source_metadata,
            >>> target_var_name, target_metadata)
            >>> conversion
            ('', '# Convert BGR to RGB\nvar1 = source_image[:, :, ::-1]\n# Change data format from HWC to CHW\nvar2 = np.transpose(var1, (2, 0, 1))\ntarget_image = var2')
        """
        conversions = self.get_conversions(source_metadata, target_metadata)
        if conversions is None:
            return None
        if len(conversions) == 0:
            return f"{target_var_name} = {source_var_name}"

        imports = set()
        main_body = []
        arg = source_var_name
        for cvt in conversions[:-1]:
            if cvt[0]:
                imports.add(cvt[0])
            return_name = f"var_{uuid.uuid4().hex}"
            main_body.append(extract_func_body(cvt[1], arg, return_name))
            arg = return_name
        if conversions[-1][0]:
            imports.add(conversions[-1][0])
        main_body.append(extract_func_body(conversions[-1][1], arg, target_var_name))

        return '\n'.join(imports), '\n'.join(main_body)

    def generate_conversion(self, source_var_name: str, source_spec, source_color_channel,
                            target_var_name: str, target_spec, target_color_channel) -> conversion:

        return self.generate_conversion_using_metadata(source_var_name, self._get_metadata(source_spec, source_color_channel),
                                                       target_var_name, self._get_metadata(target_spec, target_color_channel))

    def _get_metadata(self, spec, color_channel='color'):
        if isinstance(spec, dict):
            return spec
        return self.knowledge_graph.get_metadata_by_lib_name(spec, color_channel)
