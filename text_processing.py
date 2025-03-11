import re

def preprocess_text(text, keep_punctuation=True, custom_punctuation=" "):
    pattern = f"[{re.escape(custom_punctuation)}]"

    tokens = []
    words = re.split(r'\s+', text)
    for word in words:
        if keep_punctuation:
            split_parts = re.split(f'({pattern})', word, flags=re.IGNORECASE)
            tokens.extend([part for part in split_parts if part])
        else:
            cleaned_word = re.sub(pattern, '', word, flags=re.IGNORECASE)
            if cleaned_word:
                tokens.append(cleaned_word)
    # print(tokens)
    return tokens


def read_text_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
        # cleaned_text = re.search(r'\*\*\* START.*?\*\*\*(.*?)\*\*\* END', text, re.DOTALL)
        cleaned_text = re.search(r'(.*?)\*\*\* END', text, re.DOTALL)
        return cleaned_text.group(1).strip() if cleaned_text else "dot"


# if __name__ == "__main__":
#     text = read_text_from_file("texts\\example.txt")
#
#     custom_punctuation = ALL_PUNCTUATION
#     tokens_with_punct = preprocess_text(text, keep_punctuation=True, custom_punctuation=custom_punctuation)
#     tokens_without_punct = preprocess_text(text, keep_punctuation=False, custom_punctuation=custom_punctuation)
#
#     G_with_punct = build_network(tokens_with_punct)
#     G_without_punct = build_network(tokens_without_punct)
#
#     compare_networks(G_with_punct, G_without_punct)
#
#     plot_network(G_with_punct, "Word Network WITH Punctuation")
#     plot_network(G_without_punct, "Word Network WITHOUT Punctuation")
