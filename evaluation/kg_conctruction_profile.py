import cProfile
import pstats

from imgdataconvertcodegen.knowledge_graph_construction import (metadata_values, factories_clusters,
                                                                list_of_conversion_for_metadata_pair,
                                                                KnowledgeGraphConstructor)

if __name__ == "__main__":
    print(f"{len(factories_clusters)} factories clusters, "
          f"{sum([len(cluster[1]) for cluster in factories_clusters])} factories in total, ",
          f"{len(list_of_conversion_for_metadata_pair)} manual edge creation.")
    constructor = KnowledgeGraphConstructor(metadata_values, factories_clusters, list_of_conversion_for_metadata_pair)
    with cProfile.Profile() as profile:
        constructor.build_from_scratch()
        results = pstats.Stats(profile)
        results.sort_stats(pstats.SortKey.TIME)
        print(constructor.knowledge_graph)
        results.print_stats()
