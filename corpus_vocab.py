# Gelin Eguinosa Rosique
# 2022

from corpus_cord19 import CorpusCord19
from document_model import DocumentModel
from doc_tokenizers import doc_tokenizer
from extra_funcs import progress_bar, progress_msg


class CorpusVocab:
    """
    Manage the vocabulary of the corpus creating an id for each word, saving
    the word embeddings and the documents' vocabulary, so we can see to which
    documents each word belongs to. It also has tools to calculate the
    frequencies of the words in the documents and the corpus.
    """
    def __init__(self, corpus: CorpusCord19, doc_model: DocumentModel,
                 title_abstract=True, show_progress=False):
        """
        Create a dictionary with all the word in the corpus and their embeddings.
        Also, save the vocabulary for each document.

        Args:
            corpus: A Cord-19 Corpus class with the selection of papers in the
                corpus.
            doc_model: A Document Model class used to get the embeddings of the
                words in the corpus.
            title_abstract: Bool indicating if we will only use the title and
                abstract, or the full content of the paper.
            show_progress: A Bool representing whether we show the progress of
                the function or not.
        """
        # Create class variables.
        corpus_length = 0
        self.word2embed = {}
        self.word2count = {}
        self.doc2content = {}

        # Progress Bar Variables.
        count = 0
        total = len(corpus)

        # Iterate through the papers and extract their content.
        for cord_uid in corpus.papers_cord_uids():
            # Check if we are only using the Title & Abstract of the papers.
            if title_abstract:
                doc_content = corpus.paper_title_abstract(cord_uid)
            else:
                doc_content = corpus.paper_content(cord_uid)
            # Tokenize the content of the paper.
            doc_tokens = doc_tokenizer(doc_content)

            # Create embeddings for the new words.
            new_words = [token for token in doc_tokens if token not in self.word2embed]
            new_embeds = doc_model.words_vectors(new_words)
            for new_word, new_embed in zip(new_words, new_embeds):
                self.word2embed[new_word] = new_embed

            # Create Doc Bag-of-Words.
            doc_word2count = {}
            # Update Word count Info.
            for token in doc_tokens:
                # Update the total count of tokens in the corpus.
                corpus_length += 1
                # Update the token count in the corpus.
                if token not in self.word2count:
                    self.word2count[token] = 0
                self.word2count[token] += 1
                # Update the token count in the document.
                if token not in doc_word2count:
                    doc_word2count[token] = 0
                doc_word2count += 1

            # Save Doc Bag-of-Words.
            self.doc2content[cord_uid] = doc_word2count
            # Progress.
            if show_progress:
                count += 1
                progress_bar(count, total)
