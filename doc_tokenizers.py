# Gelin Eguinosa Rosique

from unidecode import unidecode


def doc_tokenizer(doc: str, min_len=3, max_len=15):
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
    found_tokens = []
    for doc_token in doc_tokens:
        # De-accent words and transform to Unicode characters.
        doc_token = unidecode(doc_token)

        # Remove single quotes and double quotes.
        doc_token = doc_token.replace("'", "")
        doc_token = doc_token.replace('"', '')
        doc_token = doc_token.replace('`', '')
        # Remove periods at the end of words.
        if doc_token.endswith('.'):
            doc_token = doc_token[:-1]
        # Ignore Parenthesis.
        doc_token = doc_token.replace('(', '')
        doc_token = doc_token.replace(')', '')
        # Substitute (-) with (_) to ease the recognition process.
        doc_token = doc_token.replace('-', '_')

        # Check the size of the token.
        if len(doc_token) < min_len or max_len < len(doc_token):
            continue

        # See the type of Token we have.
        if doc_token.isalpha():
            found_tokens.append(doc_token)
        elif _is_alpha_underscore(doc_token):
            found_tokens.append(doc_token)
        elif _is_covid_related(doc_token):
            found_tokens.append(doc_token)
        elif _is_numeric(doc_token):
            found_tokens.append('<numeric>')
        elif _is_coded_name(doc_token):
            found_tokens.append('<coded-name>')
        elif _is_alpha_numeric(doc_token):
            found_tokens.append('<alpha-numeric>')

    # Lower all the tokens found.
    final_tokens = [token.lower() for token in found_tokens]
    return final_tokens


def _is_alpha_underscore(word: str):
    """
    Check if 'word' is a string of the type 'e_mail', 'ex_wife', etcetera. The
    word needs to have at least a character between underscores (Ok: e_mail,
    No: e__mail).

    Args:
        word: A string with the word we want to check.

    Returns:
        True, if the word only contains alphabetical characters and underscores.
    """
    # Check the word has underscores.
    if '_' not in word:
        return False

    # Split Word by underscores and check the tokens.
    split_words = word.split('_')
    for word_section in split_words:
        if word_section.isalpha():
            continue
        else:
            # All sections need to be alphabetic.
            return False

    # The word contains only letters and underscores.
    return True


def _is_covid_related(word: str):
    """
    Check if the 'word' is a common COVID-19 term like covid_19, sars_cov_2.

    Args:
        word: A string with the word we want to check.

    Returns:
        True, if the word is a Covid term.
    """
    # Clean the word.
    lower_word = word.lower()
    lower_word = lower_word.replace('_', '')

    # Check if it is a Covid Term.
    if lower_word in {'covid19', 'sarscov2'}:
        return True
    # Check if contains something about covid-19.
    if 'covid19' in lower_word:
        return True
    # None of the above.
    return False


def _is_numeric(word: str):
    """
    Check if the given word is a number (2323, 233.11, <232,12>)
    - It can contain (.), (,), (+), (-), (*), (/), (_)

    Args:
        word: A string with the word we want to check.

    Returns:
        True, if the word represent a number, False otherwise.
    """
    for character in word:
        if character.isnumeric():
            continue
        elif character in {'.', ',', '_', '%', '-', '+'}:
            continue
        else:
            # It doesn't have recognizable numeric character.
            return False

    # It's a number.
    return True


def _is_coded_name(word: str):
    """
    Check is the string 'word' represents the coded name for an entity, virus,
    or something like that. It needs to start with a capital letter, and contain
    only letters, numbers and maybe (.) & (_). It needs to have letters between
    special characters (OK: AVI.C.D_03, No: AVI..C__D03)

    Args:
        word: A string with the word we want to check.

    Returns:
        True, if the word represents an entity or is a coded name for something.
    """
    # Check the first character is a capital letter.
    if not word[0].isupper():
        return False

    # Check the 'word' is alphanumeric.
    if not _is_alpha_numeric(word):
        return False

    # We have an entity or coded name.
    return True


def _is_alpha_numeric(word: str):
    """
    Check if the string 'word' is alphanumeric. Underscores (_) and dots (.) are
    accepted inside the word, they need to have letters between them to be
    accepted (OK: AVI.C.D_03, No: AVI..C__D03)

    Args:
        word: A string with the word we want to check.

    Returns:
        True, if we have an alphanumeric 'word'.
    """
    # Check it starts letter.
    if not word[0].isalpha():
        return False

    # Create word segments.
    word_segments = [word]

    # Check if it contains underscores (_).
    new_segments = []
    for word_section in word_segments:
        section_tokens = word_section.split('_')
        for section_token in section_tokens:
            if section_token:
                new_segments.append(section_token)
            else:
                # Empty Section between underscores.
                return False
    # Save the new Word Segments without underscores.
    word_segments = new_segments

    # Check if it contains periods (.)
    new_segments = []
    for word_section in word_segments:
        section_tokens = word_section.split('.')
        for section_token in section_tokens:
            if section_token:
                new_segments.append(section_token)
            else:
                # Empty Section between periods.
                return False
    # Save the new Word Segments without underscores.
    word_segments = new_segments

    # Check all the segments are alphanumeric.
    for word_section in word_segments:
        if word_section.isalnum():
            continue
        else:
            # Section not alphanumeric found.
            return False

    # We have an alphanumeric 'word'.
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
