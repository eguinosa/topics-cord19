# Gelin Eguinosa Rosique

from unidecode import unidecode


def doc_tokenizer(doc: str, min_len=2, max_len=15):
    """
    Tokenize the Documents from the CORD-19 corpus.
    - Lowercase tokens.
    - Ignore long and short tokens.
    - De-accent words.
    - Remove Single Quote Characters. (alzheimer's <-> alzheimers)
    - Substitute (-) with (_)
    - Substitute sentence delimiters (.),(?),(!),(;) for a <newline> meta.
    - Replace pure numbers for <num> meta.

    Args:
        doc: A String with the text we want to preprocess.
        min_len: Minimum length of token (inclusive). Shorter tokens are
            discarded.
        max_len: Maximum length of token (inclusive). Longer tokens are
            discarded.

    Returns:
        The tokens extracted from 'doc'
    """
    # Get doc's unfiltered tokens.
    doc_tokens = doc.split()

    # Filter tokens
    final_tokens = []
    for doc_token in doc_tokens:
        # Lower and De-accent words.
        doc_token = unidecode(doc_token)
        doc_token = doc_token.lower()
        # Remove single quotes and double quotes.
        doc_token = doc_token.replace("'", "")
        doc_token = doc_token.replace('"', '')
        doc_token = doc_token.replace('`', '')
        # Substitute (-) with (_) to ease the recognition process.
        doc_token = doc_token.replace('-', '_')

        # Substitute with <tag> Pure Numeric tokens.
        if _is_numeric(doc_token):
            final_tokens.append('<numeric>')
        # Alphanumeric tokens are accepted.
        elif _is_alpha_numeric(doc_token):
            # Check the size of the token.
            if min_len <= len(doc_token) <= max_len:
                final_tokens.append(doc_token)
        else:
            # Analyze all other tokens and add what the method finds.
            new_tokens = _token_analyzer(doc_token, min_len, max_len)
            final_tokens += new_tokens

    # All the tokens found.
    return final_tokens


def _token_analyzer(token_word, min_len, max_len):
    """
    Analyze tokens that contain at least a non-alphanumeric character.

    Args:
        token_word: The string we are going to analyze.
        min_len: Minimum accepted length (inclusive).
        max_len: Maximum accepted length (inclusive).

    Returns:
        A list of the tokens found inside this token-word.
    """
    # Go character by character.
    found_tokens = []
    current_token = ''
    for pos, character in enumerate(token_word):
        if character.isalnum() or character == '_':
            current_token += character
            continue

        # Some other strange character found. Split the word.
        if (min_len <= len(current_token) <= max_len
                and _is_alpha_numeric(current_token)):
            # Add word to the tokens and reset.
            found_tokens.append(current_token)
            current_token = ''

        # Add newline, if we found a sentence delimiter.
        if character in {'.', '?', '!', ';'}:
            found_tokens.append('<newline>')

    # The found tokens within the word.
    return found_tokens


def _is_numeric(word: str):
    """
    Check if the given word is a number (2323, 233.11, <232,12>)
    - It can contain (.), (,), (+), (-), (*), (/), (_)

    Args:
        word: A string with the word we want to check.

    Returns:
        True, if the word represent a number, False otherwise.
    """
    filtered_word = ''
    for character in word:
        if character.isnumeric():
            filtered_word += character
            continue
        elif character in {'.', ',', '+', '-', '*', '/', '_'}:
            continue
        else:
            # It doesn't have recognizable numeric character.
            return False

    # Check if the word without special characters is numeric.
    result = filtered_word.isnumeric()
    return result


def _is_alpha_numeric(word: str):
    """
    Check if the word is alphanumeric: A word that starts with a letter, ends
    with a letter or number, and contains only letters, numbers, and underscore
    characters. Valid words: covid19, c23, c_14. Not
    valid: 23232, c4_. It needs

    Args:
        word: The string of the word we want to recognize.

    Returns:
        A bool showing if the word is alphanumeric.
    """
    # Check it is not empty.
    if len(word) == 0:
        return False
    # Check it starts letter.
    if not word[0].isalpha():
        return False

    # Check the rest of the characters.
    for character in word[1:]:
        if character.isalnum():
            continue
        elif character == '_':
            continue
        else:
            # Not alphanumeric character found.
            return False

    # All the words after the first one passed the test.
    return True


if __name__ == '__main__':
    # Testing the Methods.
    while True:
        sentence = input("\nType a sentence to tokenize it: (q/quit to exit)\n ")
        if sentence in {'q', 'quit'}:
            break
        sentence_tokens = doc_tokenizer(sentence)
        print("\nThe tokens:")
        print(sentence_tokens)
