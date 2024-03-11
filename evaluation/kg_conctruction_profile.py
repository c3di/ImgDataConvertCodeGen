import cProfile
import pstats

from imgdataconvertcodegen.knowledge_graph_construction import (metadata_values, all_edge_factories,
                                                                KnowledgeGraphConstructor)

if __name__ == "__main__":
    constructor = KnowledgeGraphConstructor(metadata_values, all_edge_factories)
    with cProfile.Profile() as profile:
        constructor.build_from_scratch()
        print(constructor.knowledge_graph)
        results = pstats.Stats(profile)
        results.sort_stats(pstats.SortKey.TIME)
        results.print_stats()
