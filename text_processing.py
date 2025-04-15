import re
from config_setter import load_config


def tokenize_text(text, keep_punctuation=True):
    config = load_config()
    pattern = f"[{re.escape(config["punctuation_pattern"])}]+"
    mode = config["punctuation_mode"]

    tokens = []
    words = re.split(r'(\s+)', text)

    for word in words:
        if keep_punctuation:
            split_parts = re.split(f'({pattern})', word, flags=re.IGNORECASE)
            merged_tokens = []

            for part in split_parts:
                if part and re.fullmatch(pattern, part):
                    if mode == "merged":
                        if merged_tokens and re.fullmatch(pattern, merged_tokens[-1]):
                            merged_tokens[-1] += part
                        else:
                            merged_tokens.append(part)
                    elif mode == "single":
                        merged_tokens.append(".")
                    elif mode == "separate":
                        merged_tokens.append(part)
                elif part.strip():
                    merged_tokens.append(part)

            tokens.extend(merged_tokens)
        else:
            cleaned_word = re.sub(pattern, '', word, flags=re.IGNORECASE)
            if cleaned_word.strip():
                tokens.append(cleaned_word.strip())

    return tokens


def sentence_tokenize_text(text):
    config = load_config()
    sentence_delimiters = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    sentences = re.split(sentence_delimiters, text)
    print(sentences)
    return [s.strip() for s in sentences if s.strip()]


def read_text_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
        return text
