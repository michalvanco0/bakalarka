import networkx as nx
import numpy as np
from scipy.stats import linregress


def jaccard_similarity(G1, G2):
    edges1, edges2 = set(G1.edges()), set(G2.edges())
    intersection = len(edges1 & edges2)
    union = len(edges1 | edges2)
    # return nx.jaccard_coefficient(G1, G2) if union else 0
    return intersection / union if union else 0


# list
def degree_distribution(G):
    return sorted([(d, i) for i, d in G.degree()], reverse=True)


def graph_density(G):
    return nx.density(G)


def average_path_length(G):
    if nx.is_weakly_connected(G):
        return nx.average_shortest_path_length(G)
    return float('inf')


def clustering_coefficient(G):
    return nx.average_clustering(G.to_undirected())


def log_bin_degrees(degrees, num_bins=10):
    min_degree = min(degrees)
    max_degree = max(degrees)

    bins = np.logspace(np.log10(min_degree), np.log10(max_degree), num_bins)
    hist, edges = np.histogram(degrees, bins=bins, density=True)

    bin_centers = np.sqrt(edges[:-1] * edges[1:])
    return bin_centers, hist


def compute_slope(x, y):
    log_x = np.log10(x[y > 0])
    log_y = np.log10(y[y > 0])

    slope, intercept, r_value, _, _ = linregress(log_x, log_y)
    print(slope, intercept)
    return slope, intercept


def compare_networks(G1, G2):
    result = f'''
    Network with Punctuation
        Nodes: {len(G1.nodes())}, Edges: {len(G1.edges())}
        Density: {graph_density(G1):.4f}
        Clustering Coefficient: {clustering_coefficient(G1):.4f}
        Degree Distribution: {degree_distribution(G1)[:10]}
        Slope:
        
    Network without Punctuation
        Nodes: {len(G2.nodes())}, Edges: {len(G2.edges())}
        Density: {graph_density(G2):.4f}
        Clustering Coefficient: {clustering_coefficient(G2):.4f}
        Degree Distribution: {degree_distribution(G2)[:10]}
        Slope:
    
    Jaccard Similarity (Edges): {jaccard_similarity(G1, G2):.4f}
    '''
    return result
