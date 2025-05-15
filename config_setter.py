import json
import os

ALL_PUNCTUATION = "!\"\'()*,—./’“”:;<=>?[]_`{|}~»«..."
BASIC = '.?!,'
CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {
    "punctuation_mode": "split",
    "punctuation_pattern": "!\"\'()*,—./’“”:;<=>?[]_`{|}~»«...",
    "save_directory": "C:\\Users\\User\\bakalarka\\bakalarka\\analysis_results",
    "node_color": "black",
    "edge_color": "gray",
    "punc_node_color": "red",
    "punc_edge_color": "salmon",
    "weighed": False,
    "directed": False,
    "show_labels": False,
    "show_net": False,
    "min_degree": 0,
    "show_histogram": False,
    "show_fit_convergence": False,
    "show_weibull": False,
    "show_distribution_comparison": False,
    "show_models": False,
    "show_binned": False
}


def load_config():
    with open(CONFIG_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def save_config(config):
    with open(CONFIG_FILE, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=4)


def update_config(key, value):
    config = load_config()
    config[key] = value
    save_config(config)
