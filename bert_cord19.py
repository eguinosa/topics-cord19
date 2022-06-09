# Gelin Eguinosa Rosique

from os import mkdir
from os.path import isdir, join

from sentence_transformers import SentenceTransformer, util

from time_keeper import TimeKeeper


class BertCord19:
    """
    Load and Manage the BERT pre-trained models to the get the vector
    representations of the words and documents in the CORD-19 corpus.
    The models are available in:
    'https://www.sbert.net/docs/pretrained_models.html'
    """
    # Data Locations
    models_folder = 'doc_models'

    # Bert Models used.
    models_dict = {
        'all-mpnet-base-v2': {
            'similarity': 'dot_score',
            'max_seq_length': 384,
            'dimensions': 768,
            'performance': 63.30,
            'speed': 2_800,
            'size': 420,
        },
        'all-MiniLM-L12-v2': {
            'similarity': 'dot_score',
            'max_seq_length': 256,
            'dimensions': 384,
            'performance': 59.76,
            'speed': 7_500,
            'size': 120,
        },
        'all-MiniLM-L6-v2': {
            'similarity': 'dot_score',
            'max_seq_length': 256,
            'dimensions': 384,
            'performance': 58.80,
            'speed': 14_200,
            'size': 80,
        },
        'paraphrase-MiniLM-L3-v2': {
            'similarity': 'cos_sim',
            'max_seq_length': 128,
            'dimensions': 384,
            'performance': 50.74,
            'speed': 19_000,
            'size': 61,
        },
        'average_word_embeddings_glove.6B.300d': {
            'similarity': 'cos_sim',
            'max_seq_length': -1,
            'dimensions': 300,
            'performance': 36.25,
            'speed': 34_000,
            'size': 420,
        },
    }

    def __init__(self, model_name, show_progress=False):
        """
        Load or Download the BERT model we are going to use to model the words
        and documents.

        Args:
            model_name: A String with the name of model.
            show_progress: Bool representing whether we show the progress of
                the function or not.
        """
        # Check if we support the Bert Model Requested.
        if model_name not in self.models_dict:
            raise NameError("BertCord19() doesn't support the requested model.")
        # Save the Model's Attributes.
        self.model_name = model_name
        self.model_info = self.models_dict[model_name]

        # Check the Models' Folder exists.
        if not isdir(self.models_folder):
            mkdir(self.models_folder)

        # Load or Download Model.
        self.model_path = join(self.models_folder, self.model_name)
        if isdir(self.model_path):
            # The Model is available locally.
            self.model = SentenceTransformer(self.model_path)
            if show_progress:
                print(f"The model <{self.model_name}> loaded successfully.")
        else:
            # Download model.
            self.model = SentenceTransformer(f'sentence-transformers/{self.model_name}')
            # Save model locally.
            self.model.save(self.model_path, self.model_name, create_model_card=True)
            if show_progress:
                print(f"The model <{self.model_name}> downloaded and saved.")

    def word2vec(self, word):
        """
        Transform a word into a vector.

        Args:
            word: a string of one token.

        Returns:
            The vector of the word.
        """
        result = self.model.encode(word)
        return result

    def words2vec(self, words):
        """
        Transform a list of words into their vector representation (tensor).

        Args:
            words: List of strings containing the words.

        Returns:
            A list of word vectors.
        """
        result = self.model.encode(words)
        return result

    def doc2vec(self, doc_text):
        """
        Transform the text of a document into a vector.

        Args:
            doc_text: A string containing the text of the document.

        Returns:
            The vector of the document.
        """
        result = self.model.encode(doc_text)
        return result

    def docs2vecs(self, docs_texts):
        """
        Transform the text of the documents to their vector representation (tensor).

        Args:
            docs_texts: A list of strings with the content of the documents.

        Returns:
            A list with the tensor vector of the documents.
        """
        result = self.model.encode(docs_texts)
        return result


def similarity(text1, text2):
    """
    Find the similarity score between the 2 documents.

    Args:
        text1: The tensor or text of the first document.
        text2: The tensor or text of the second document.

    Returns:
        Their similarity score.
    """
    result = util.cos_sim(text1, text2)

    return result


if __name__ == '__main__':
    # Record the Runtime of the Program
    stopwatch = TimeKeeper()

    # Creating Model.
    print("\nCreating Bert Model...")
    my_model = BertCord19('average_word_embeddings_glove.6B.300d', show_progress=True)
    print("Done.")
    print(f"[{stopwatch.formatted_runtime()}]")

    print("\nTesting word similarities (To close use [q/quit]):")
    quit_words = {'q', 'quit'}
    while True:
        word1 = input("Type the first word: ")
        if word1 in quit_words:
            break
        word2 = input("Type the second word: ")
        if word2 in quit_words:
            break
        sim_words = util.cos_sim(my_model.word2vec(word1), my_model.word2vec(word2))[0][0]
        print("The words similarity:")
        print(sim_words)

    print("\nDone.")
    print(f"[{stopwatch.formatted_runtime()}]\n")