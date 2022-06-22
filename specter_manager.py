# Gelin Eguinosa Rosique

import numpy as np

from corpus_cord19 import CorpusCord19
from document_model import DocumentModel
from papers import Papers
from cord19_utilities import load_vocab_embeddings, embeds_title_abstract
from time_keeper import TimeKeeper


class SpecterManager(DocumentModel):
    """
    Load & Manage the saved embeddings of the words and documents in the CORD-19
    dataset.
    """

    def __init__(self, cord19_papers: CorpusCord19 = None, vocab_embeddings=None,
                 show_progress=False):
        """
        Load and save an instance of the class Papers() to get access to the
        papers embeddings in CORD-19, and also get the specter embeddings for
        the vocabulary of the content we are going to use.

        By default, if no 'cord19_papers' or 'vocab_embeddings' is provided,
        creates and loads its own Papers() class and the vocabulary embeddings
        for words in the Titles and Abstracts of the CORD-19 dataset.

        Args:
            cord19_papers: Papers() class used to access the specter embeddings
                of the papers.
            vocab_embeddings: Dictionary containing the specter embeddings of
                the words in the vocabulary we are going to use.
            show_progress: Bool representing whether we show the progress of
                the function or not.
        """
        # Load the Papers() class.
        if show_progress:
            print("Loading Papers' embeddings...")
        if cord19_papers:
            self.corpus = cord19_papers
        else:
            self.corpus = Papers(show_progress=show_progress)

        # Load the vocabulary embeddings.
        if show_progress:
            print("Loading Vocabulary's embeddings...")
        if vocab_embeddings:
            self.vocab_embeds = vocab_embeddings
        else:
            self.vocab_embeds = load_vocab_embeddings(embeds_title_abstract)

    def model_type(self):
        """
        Give the type of the document model that was used to create the
        embeddings.

        Returns: String with the name of the model 'specter'.
        """
        return 'specter'

    def word_vector(self, word):
        """
        Use the embeddings' dictionary to find the specter vector for the given
        'word'. If the word is not found in the vocabulary, then return a Zero
        vector.

        Args:
            word: String containing the word.

        Returns:
            Numpy.ndarray containing the embedding of the word.
        """
        # See if the word is in the vocabulary.
        if word in self.vocab_embeds:
            word_embed = self.vocab_embeds[word]
            numpy_embed = np.array(word_embed)
        else:
            numpy_embed = np.array([0])

        # The specter embedding of the word.
        return numpy_embed

    def document_vector(self, doc_cord_uid):
        """
        Use the CORD-19 dataset to find the specter embedding for the given
        'doc_cord_uid' paper.

        Args:
            doc_cord_uid: String with the ID of the paper we want to find the
                embedding.

        Returns:
            Numpy.ndarray with the specter embedding of the document.
        """
        paper_specter = self.corpus.paper_embedding(doc_cord_uid)
        numpy_embed = np.array(paper_specter)
        return numpy_embed


if __name__ == '__main__':
    # Record Runtime of the Program.
    stopwatch = TimeKeeper()

    # Test Specter Manager.
    print("\nLoading Specter Manager...")
    the_manager = SpecterManager(show_progress=True)
    print("Done.")
    print(f"[{stopwatch.formatted_runtime()}]")

    # Get words embeddings:
    while True:
        input_word = input("\nType a word from the vocabulary: ")
        if input_word in {'', 'q', 'quit'}:
            break
        input_embed = the_manager.word_vector(input_word)
        print(f"Specter embedding of <{input_word}>:")
        print(input_embed)
        print(f"Type: {type(input_embed)}")

    # Get Paper's embeddings:
    while True:
        input_id = input("\nType the cord_uid of the paper: ")
        if input_id in {'', 'q', 'quit'}:
            break
        paper_embed = the_manager.document_vector(input_id)
        print(f"Specter embedding of paper <{input_id}>:")
        print(paper_embed)
        print(f"Type: {type(paper_embed)}")

    print("Done.")
    print(f"[{stopwatch.formatted_runtime()}]\n")
