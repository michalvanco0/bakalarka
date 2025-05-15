import re
from collections import Counter
import networkx as nx
import nltk
import numpy as np
from scipy.optimize import minimize
from scipy.stats import linregress, poisson
from textstat import textstat
from nltk.probability import FreqDist
from nltk import pos_tag, ne_chunk, BigramCollocationFinder, BigramAssocMeasures
from nltk.tree import Tree
from textblob import TextBlob
from config_setter import load_config
from text_processing import tokenize_text, sentence_tokenize_text

# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('maxent_ne_chunker_tab')
# nltk.download('punkt')
# nltk.download('punkt_tab')


def jaccard_similarity(G1, G2):
    edges1, edges2 = set(G1.edges()), set(G2.edges())
    intersection = len(edges1 & edges2)
    union = len(edges1 | edges2)
    # return nx.jaccard_coefficient(G1, G2) if union else 0
    return intersection / union if union else 0


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
    degrees = np.array(degrees)
    degrees = degrees[degrees > 0]

    min_degree = degrees.min()
    max_degree = degrees.max()

    bins = np.logspace(np.log10(min_degree), np.log10(max_degree), num_bins + 1)
    hist, edges = np.histogram(degrees, bins=bins, density=True)

    bin_centers = np.sqrt(edges[:-1] * edges[1:])
    return bin_centers, hist


def compute_slope(x, y):
    log_x = np.log10(x[y > 0])
    log_y = np.log10(y[y > 0])
    coeffs = np.polyfit(log_x, log_y, 1)
    return coeffs[0], coeffs[1]


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


def get_punctuation_distances(tokens, punctuation_set):
    distances = []
    last_index = None
    for i, token in enumerate(tokens):
        if token in punctuation_set:
            if last_index is not None:
                dist = i - last_index - 1
                if dist > 0:
                    distances.append(dist)
            last_index = i
    return distances


def weibull_pmf(k, q, beta):
    return q ** (k ** beta) - q ** ((k + 1) ** beta)


def poisson_lambda(ks, freqs):
    return np.sum(ks * freqs)


def geometric_p(ks, freqs):
    return 1 / (np.sum(ks * freqs) + 1)


def weibull_fit(distances):
    if not distances:
        return None, None, None

    counts = Counter(distances)
    ks = np.array(sorted(counts.keys()))
    freqs = np.array([counts[k] for k in ks], dtype=float)
    freqs /= freqs.sum()

    def loss(params):
        q, beta = params
        if not (0 < q < 1) or beta <= 0:
            return np.inf
        model = weibull_pmf(ks, q, beta)
        return np.sum((freqs - model) ** 2)

    result = minimize(loss, [0.5, 1.0], bounds=[(1e-5, 1-1e-5), (1e-2, 10)])
    if result.success:
        q_fit, beta_fit = result.x
        return ks, freqs, q_fit, beta_fit
    else:
        return ks, freqs, None, None


def get_weibull_parameters(tokens, punctuation_set):
    distances = get_punctuation_distances(tokens, punctuation_set)
    ks, freqs, q_fit, beta_fit = weibull_fit(distances)
    return ks, freqs, q_fit, beta_fit


def ling_analysis(text, slopes=None):
    punc_pattern = load_config()["punctuation_pattern"]
    tokens = tokenize_text(text, False)
    tokens_with = tokenize_text(text, True)
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

    ks, freqs, q, beta = get_weibull_parameters(tokens_with, punc_pattern)

    stats_dict = {
        "Sentences count": sentences_count,
        "Average sentence length": avg_sentence_length(sentences),
        "Word count": len(tokens),
        "Unique word count": len(set(tokens)),
        "Punctuation count": punctuation_count,
        "Punctuation density": punctuation_count / len(text),
        "Character count": len(text),
        "Word frequency (top 10)": freq(tokens),
        "Sentiment": "Positive" if sent > 0 else "Negative" if sent < 0 else "Neutral",
        "Readability": readability(text),
        "Syllable count": textstat.syllable_count(text),
        "POS distribution": pos_distribution(tokens),
        "Repetition index": repetition_index(tokens),
        "Punctuation patterns": punctuation_patterns,
        "Inter-punctuation distance avg": avg_dist,
        "Inter-punctuation distance min": min_dist,
        "Inter-punctuation distance max": max_dist,
        "Punctuation per sentence avg": avg_punct_per_sentence,
        "Terminal punctuation frequency": dict(terminal_counts),
        "Paragraph count": len(paragraphs),
        "Average paragraph length (words)": np.mean([len(p.split()) for p in paragraphs]) if paragraphs else 0,
        "Power law slope": power_law_slope,
        "Poisson lambda": poisson_lambda,
        "p": 1-q,
        "beta": beta,
    }
    if slopes:
        stats_dict.update(slopes)
    table_data = [["Metric", "Value"]]
    for key, value in stats_dict.items():
        table_data.append([key, str(value)])
    return table_data


def compare_networks(G1, G2):
    G1_nodes = list(G1.nodes())
    G2_nodes = list(G2.nodes())
    table_data = [
        ["Metric", "With Punctuation", "Without Punctuation"],
        ["Nodes", len(G1.nodes()), len(G2.nodes())],
        ["Edges", len(G1.edges()), len(G2.edges())],
        ["Density", f"{graph_density(G1):.4f}", f"{graph_density(G2):.4f}"],
        ["Clustering Coefficient", f"{clustering_coefficient(G1):.4f}", f"{clustering_coefficient(G2):.4f}"],
        ["Degree Distribution (top 10)", str(degree_distribution(G1)[:10]), str(degree_distribution(G2)[:10])],
        ["Slope", f"{get_slope(G1)[0]:.4f}", f"{get_slope(G2)[0]:.4f}"],
        ["Collocations", str(extract_collocations(G1_nodes, 10)), str(extract_collocations(G2_nodes, 10))],
        # ["Lexical Diversity", f"{lexical_diversity(G1_nodes):.4f}", f"{lexical_diversity(G2_nodes):.4f}"],
        ["Jaccard Similarity (Edges)", f"{jaccard_similarity(G1, G2):.4f}", ""],
    ]

    return table_data
