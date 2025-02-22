def estimate_tokens(text):
    # Rough estimation based on common tokenization patterns
    # 1 token ~= 4 characters for English text
    char_count = len(text)
    word_count = len(text.split())

    # Use the larger of the two estimates
    char_based = char_count / 4
    word_based = word_count * 1.3

    return max(int(char_based), int(word_based))
