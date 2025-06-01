import re
from config_setter import load_config, ALL_PUNCTUATION


def tokenize_text(text, keep_punctuation=True):
    config = load_config()
    pattern = f"[{re.escape(config['punctuation_pattern'])}]+"
    pattern_all = f"[{re.escape(ALL_PUNCTUATION)}]+"
    mode = config["punctuation_mode"]

    tokens = []
    words = re.split(r'(\s+)', text)

    for word in words:
        if keep_punctuation:
            split_parts = re.split(f'({pattern})', word, flags=re.IGNORECASE)
            merged_tokens = []

            for part in split_parts:
                if part and re.fullmatch(pattern, part):
                    smart_parts = split_punctuation(part)
                    for sp in smart_parts:
                        if mode == "merged":
                            merged_tokens.append(sp)
                        elif mode == "single":
                            merged_tokens.append(".")
                        elif mode == "separate":
                            merged_tokens.extend(list(sp))
                elif part.strip():
                    merged_tokens.append(part)

            tokens.extend(merged_tokens)
        else:
            cleaned_word = re.sub(pattern_all, '', word, flags=re.IGNORECASE)
            if cleaned_word.strip():
                tokens.append(cleaned_word.strip())

    return tokens


def split_punctuation(punct_string):
    groups = []
    current = punct_string[0]

    for ch in punct_string[1:]:
        if ch == current[-1]:
            current += ch
        else:
            groups.append(current)
            current = ch
    groups.append(current)
    return groups


def sentence_tokenize_text(text):
    sentence_delimiters = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    sentences = re.split(sentence_delimiters, text)
    return [s.strip() for s in sentences if s.strip()]


def read_text_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
        return text
