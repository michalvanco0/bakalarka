import json
import os

ALL_PUNCTUATION = "!\"\'()*+,-./:;<=>?[]^_`{|}~»«"
CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {
    "punctuation_mode": "split",
    "punctuation_pattern": "!\"\'()*+,-./:;<=>?[]^_`{|}~»«",
    "save_directory": "C:\\Users\\User\\bakalarka\\bakalarka\\analysis_results",
    "node_color": "black",
    "edge_color": "gray",
    "punc_node_color": "red",
    "punc_edge_color": "salmon",
    "weighed": False,
    "directed": False,
    "show_labels": False
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
