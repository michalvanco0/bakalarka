import re
from collections import Counter

import networkx as nx
import nltk
import numpy as np
import spacy
from scipy.stats import linregress, poisson
from textstat import textstat
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
from nltk import pos_tag, ne_chunk, BigramCollocationFinder, BigramAssocMeasures
from nltk.tree import Tree
from textblob import TextBlob

from config_setter import load_config
from text_processing import tokenize_text, sentence_tokenize_text

nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')
nltk.download('punkt')
nltk.download('punkt_tab')
ALL_PUNCTUATION = '!"\'()*+,-./:;<=>?[\\]^_`{|}~»«'

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


def get_degrees(G):
    return [deg for _, deg in G.degree()]


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
    # print(slope, intercept)
    return slope, intercept


def get_slope(G):
    degrees = get_degrees(G)
    bin_centers, hist = log_bin_degrees(degrees)
    return compute_slope(bin_centers, hist)


def freq(tokens):
    word_freq = Counter(tokens)
    return dict(word_freq.most_common(10))


def ner(tokens):
    try:
        pos_tags = pos_tag(tokens)
        named_entities = []
        for chunk in ne_chunk(pos_tags):
            if isinstance(chunk, Tree):
                named_entity = " ".join(c[0] for c in chunk)
                named_entities.append(named_entity)
        return named_entities
    except Exception as e:
        print(f"An error occurred: {e}")
        return set()


def sentiment(text):
    return TextBlob(text).sentiment.polarity


def readability(text):
    return textstat.flesch_reading_ease(text)


def compute_power_law(punctuation_counts):
    freqs = np.array(list(punctuation_counts.values()))
    ranks = np.arange(1, len(freqs) + 1)
    log_freqs = np.log10(freqs)
    log_ranks = np.log10(ranks)
    slope, intercept, _, _, _ = linregress(log_ranks, log_freqs)
    return slope, intercept


def compute_poisson_fit(punctuation_counts, sentence_count):
    lambda_ = sum(punctuation_counts.values()) / sentence_count
    expected_counts = {k: poisson.pmf(k, lambda_) for k in range(max(punctuation_counts.values()) + 1)}
    return lambda_, expected_counts


def extract_collocations(tokens, top_n=10):
    finder = BigramCollocationFinder.from_words(tokens)
    scored = finder.score_ngrams(BigramAssocMeasures.pmi)
    return scored[:top_n]


def lexical_diversity(tokens):
    return len(set(tokens)) / len(tokens) if tokens else 0


def avg_sentence_length(sentences):
    return np.mean([len(tokenize_text(s, False)) for s in sentences])


def pos_distribution(tokens):
    tags = nltk.pos_tag(tokens)
    pos_counts = Counter(tag for word, tag in tags)
    return dict(pos_counts)


def repetition_index(tokens):
    freq = FreqDist(tokens)
    return sum(freq[word] > 1 for word in freq) / len(freq)


def ling_analysis(text):
    punc_pattern = load_config()["punctuation_pattern"]
    tokens = tokenize_text(text, False)
    sent = sentiment(text)
    sentences = sentence_tokenize_text(text)
    sentences_count = len(sentences)
    punctuation_count = sum(1 for char in text if char in punc_pattern)
    escaped_pattern = re.escape(punc_pattern)
    regex_pattern = r'[' + escaped_pattern + r']'
    punctuation_patterns = Counter(re.findall(regex_pattern, text))
    power_law_slope, power_law_intercept = compute_power_law(punctuation_patterns)
    poisson_lambda, poisson_fit = compute_poisson_fit(punctuation_patterns, sentences_count)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    terminal_counts = Counter(s.strip()[-1] for s in sentences if s.strip()[-1] in ".!?")
    punct_indices = [i for i, c in enumerate(text) if c in punc_pattern]
    inter_punct_distances = [j - i for i, j in zip(punct_indices[:-1], punct_indices[1:])]
    if inter_punct_distances:
        avg_dist = np.mean(inter_punct_distances)
        min_dist = np.min(inter_punct_distances)
        max_dist = np.max(inter_punct_distances)
    else:
        avg_dist = min_dist = max_dist = 0

    punct_per_sentence = [sum(1 for char in s if char in punc_pattern) for s in sentences]
    avg_punct_per_sentence = np.mean(punct_per_sentence) if punct_per_sentence else 0

    result = f'''
    Sentences count: {sentences_count}
    Average sentence length: {avg_sentence_length(sentences)}
    Word count: {len(tokens)}
    Unique word count: {len(set(tokens))}
    Punctuation count = {punctuation_count}
    Punctuation density = {punctuation_count / len(text)}
    Character count: {len(text)}
    Word frequency (10): {freq(tokens)}
    Named entities: {ner(tokens)}
    Sentiment: {"Positive" if sent > 0 else "Negative" if sent < 0 else "Neutral"}
    Readability: {readability(text)}
    Syllable count: {textstat.syllable_count(text)}
    Pos distribution: {pos_distribution(tokens)}
    Repetition index: {repetition_index(tokens)}
    punctuation_patterns: {punctuation_patterns}
    inter_punctuation_distance_avg: {avg_dist}
    inter_punctuation_distance_min: {min_dist}
    inter_punctuation_distance_max: {max_dist}
    punctuation_per_sentence_avg: {avg_punct_per_sentence}
    terminal_punctuation_frequency: {dict(terminal_counts)}
    paragraph_count: {len(paragraphs)}
    avg_paragraph_length: {np.mean([len(p.split()) for p in paragraphs]) if paragraphs else 0} words
       
    
    Power law slope: {power_law_slope}
    Poisson lambda: {poisson_lambda}
    '''

    return result


def compare_networks(G1, G2):
    G1_nodes = list(G1.nodes())
    G2_nodes = list(G2.nodes())
    result = f'''
    Network with Punctuation
        Nodes: {len(G1.nodes())}
        Edges: {len(G1.edges())}
        Density: {graph_density(G1):.4f}
        Clustering Coefficient: {clustering_coefficient(G1):.4f}
        Degree Distribution: {degree_distribution(G1)[:10]}
        Slope: {get_slope(G1)[0]}
        Collocations: {extract_collocations(G1_nodes, 10)}
        Lexical diversity: {lexical_diversity(G1_nodes):.4f}
        
    Network without Punctuation
        Nodes: {len(G2.nodes())}
        Edges: {len(G2.edges())}
        Density: {graph_density(G2):.4f}
        Clustering Coefficient: {clustering_coefficient(G2):.4f}
        Degree Distribution: {degree_distribution(G2)[:10]}
        Slope: {get_slope(G2)[0]}
        Collocations: {extract_collocations(G2_nodes, 10)}
        Lexical diversity: {lexical_diversity(G2_nodes):.4f}
    
    Jaccard Similarity (Edges): {jaccard_similarity(G1, G2):.4f}
    '''

    return result
