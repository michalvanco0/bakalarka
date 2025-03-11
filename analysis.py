import networkx as nx


def jaccard_similarity(G1, G2):
    edges1, edges2 = set(G1.edges()), set(G2.edges())
    intersection = len(edges1 & edges2)
    union = len(edges1 | edges2)
    # return nx.jaccard_coefficient(G1, G2) if union else 0
    return intersection / union if union else 0


# list
def degree_distribution(G):
    return sorted([d for _, d in G.degree()], reverse=True)


def graph_density(G):
    return nx.density(G)


def average_path_length(G):
    if nx.is_weakly_connected(G):
        return nx.average_shortest_path_length(G)
    return float('inf')


def clustering_coefficient(G):
    return nx.average_clustering(G.to_undirected())


def compare_networks(G1, G2):
    result = f'''
    Network with Punctuation
        Nodes: {len(G1.nodes())}, Edges: {len(G1.edges())}
        Density: {graph_density(G1):.4f}
        Clustering Coefficient: {clustering_coefficient(G1):.4f}

    Network without Punctuation
        Nodes: {len(G2.nodes())}, Edges: {len(G2.edges())}
        Density: {graph_density(G2):.4f}
        Clustering Coefficient: {clustering_coefficient(G2):.4f}
    
    Jaccard Similarity (Edges): {jaccard_similarity(G1, G2):.4f}
    '''
    return result
