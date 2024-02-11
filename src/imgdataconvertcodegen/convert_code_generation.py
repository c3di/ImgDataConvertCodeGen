import uuid

from src.imgdataconvertcodegen.function_util import create_unique_function, extract_func_body
from src.imgdataconvertcodegen.knowledge_graph_construction import encode_to_string


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

    def conversion_functions(self, source_metadata, target_metadata) -> list[str] | None:
        source_encode_str = encode_to_string(source_metadata)
        target_encode_str = encode_to_string(target_metadata)
        if (source_encode_str, target_encode_str) in self._cache:
            return self._cache[(source_encode_str, target_encode_str)]
        path = self.knowledge_graph.get_shortest_path(source_metadata, target_metadata)
        if path is None:
            return None
        functions = []
        for i in range(len(path) - 1):
            edge = self.knowledge_graph.get_edge(path[i], path[i + 1])
            functions.append(edge['conversion'])
        self._cache[(source_encode_str, target_encode_str)] = functions
        return functions

    def generate_code_using_metadata(self, source_var_name, source_metadata,
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
            >>> code = convert_code_generator.generate_code_using_metadata(source_var_name, source_metadata,
            >>> target_var_name, target_metadata)
            >>> print(code)
            # Convert BGR to RGB
            var1 = source_image[:, :, ::-1]
            # Change data format from HWC to CHW
            var2 = np.transpose(var1, (2, 0, 1))
            target_image = var2
        """
        functions = self.conversion_functions(source_metadata, target_metadata)
        if functions is None:
            return None
        code_snippets = []
        arg = source_var_name
        for function in functions:
            return_name = f"var_{uuid.uuid4().hex}"
            code_snippets.append(extract_func_body(function, arg, return_name))
            arg = return_name
        code_snippets.append(f"{target_var_name} = {arg}")
        return '\n'.join(code_snippets)

    def generate_code(self, source_var_name: str, source_spec, target_var_name: str, target_spec) -> str | None:

        return self.generate_code_using_metadata(source_var_name, self._get_metadata(source_spec),
                                                 target_var_name, self._get_metadata(target_spec))

    def _get_metadata(self, spec):
        if isinstance(spec, dict):
            return spec
        return self.knowledge_graph.get_metadata_by_lib_name(spec)
