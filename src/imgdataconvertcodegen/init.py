from src.imgdataconvertcodegen.knowledge_graph_construction import create_knowledge_graph
from src.imgdataconvertcodegen.convert_code_generation import ConvertCodeGenerator

knowledge_graph = create_knowledge_graph()
code_generator = ConvertCodeGenerator(knowledge_graph)


def get_covert_code(source_var_name: str, source_metadata, target_var_name, target_metadata):
    return code_generator.generate_code_without_intermediate_func(source_var_name, source_metadata,
                                                                  target_var_name, target_metadata)
