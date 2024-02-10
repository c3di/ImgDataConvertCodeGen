from src.imgdataconvertcodegen.function_util import create_unique_function


class ConvertCodeGenerator:
    _knowledge_graph = None

    def __init__(self, knowledge_graph):
        self._knowledge_graph = knowledge_graph

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
        path = self.knowledge_graph.get_shortest_path(source_metadata, target_metadata)
        if path is None:
            return None
        functions = []
        for i in range(len(path) - 1):
            edge = self.knowledge_graph.get_edge(path[i], path[i + 1])
            functions.append(edge['conversion'])
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

        Returns:
            str: A string containing the Python code necessary to perform the conversion. This code includes
             the definitions of one or more conversion functions and a final line that applies these functions
             in sequence to achieve the desired transformation.

        Examples:
            >>> source_var_name = "source_image"
            >>> source_metadata = {"color_channel": "bgr", "channel_order": "channel last", ...}
            >>> target_var_name = "target_image"
            >>> target_metadata = {"color_channel": "rgb", "channel_order": "channel first", ...}
            >>> convert_code_generator = ConvertCodeGenerator()
            >>> code = convert_code_generator.generate_code_using_metadata(source_var_name, source_metadata,
            >>> target_var_name, target_metadata)
            >>> print(code)
            def convert_1(var):
                # Convert BGR to RGB
                return var[:, :, ::-1]
            def convert_2(var):
                # Change data format from HWC to CHW
                return np.transpose(var, (2, 0, 1))
            target_image = convert_2(convert_1(source_image))

        Note:
            The actual implementation of the conversion functions (`convert_1`, `convert_2`, etc.) and their sequence
             will vary depending on the specifics of the `source_metadata` and `target_metadata`.

        """
        functions = self.conversion_functions(source_metadata, target_metadata)
        if functions is None:
            return None
        definitions = []
        code = source_var_name
        for function in functions:
            unique_func = create_unique_function(function)
            definitions.append(unique_func['function_definition'])
            code = f"{unique_func['function_name']}({code})"
        return f'{"\n".join(definitions)}\n{target_var_name} = {code}'

    def generate_code(self, source_var_name: str, source_spec, target_var_name: str, target_spec) -> str | None:

        return self.generate_code_using_metadata(source_var_name, self._get_metadata(source_spec),
                                                 target_var_name, self._get_metadata(target_spec))

    def _get_metadata(self, spec):
        if isinstance(spec, dict):
            return spec
        return self.knowledge_graph.get_metadata_by_lib_name(spec)
