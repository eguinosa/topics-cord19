# Gelin Eguinosa Rosique
# 2022

from gensim.models import doc2vec

from doc_tokenizers import doc_tokenizer


class IterableTokenizer:
    """
    Create an iterable class containing the tokenized documents of a corpus.
    """

    def __init__(self, docs_generator_func, tagged_tokens=False):
        """
        Save the function that generates the documents of the corpus, and also
        if the documents need to be tagged or not.

        Args:
            docs_generator_func: The function we are calling to get the documents
                in the corpus.
            tagged_tokens: A bool telling us if we need to tag the documents.
        """
        self.docs_generator_func = docs_generator_func
        self.tagged_tokens = tagged_tokens

    def __iter__(self):
        """
        Create an iterable sequence of containing the tokens of the documents in
        the corpus. The documents can be tagged, depending on the value of the
        'self.tagged_tokens' attributes.

        Returns: A lazy sequence containing the list of tokens (maybe tagged) in
            corpus.
        """
        # Get the documents we are going to tokenize.
        documents = self.docs_generator_func()
        # Generate the documents with their order as the tag.
        for tag, doc in enumerate(documents):
            # Tokenize the documents.
            doc_tokens = doc_tokenizer(doc)
            if self.tagged_tokens:
                yield doc2vec.TaggedDocument(doc_tokens, [tag])
            else:
                yield doc_tokens
