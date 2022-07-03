# Gelin Eguinosa Rosique

from os import mkdir
from os.path import isdir, join
from sentence_transformers import SentenceTransformer, util

from document_model import DocumentModel
from time_keeper import TimeKeeper


class BertCord19(DocumentModel):
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
        # English Models.
        'all-mpnet-base-v2': {
            'type': 'bert-english',
            'max_seq_length': 384,
            'dimensions': 768,
            'performance': 63.30,
            'speed': 2_800,
            'size': 420,
        },
        'all-MiniLM-L12-v2': {
            'type': 'bert-english',
            'max_seq_length': 256,
            'dimensions': 384,
            'performance': 59.76,
            'speed': 7_500,
            'size': 120,
        },
        'all-MiniLM-L6-v2': {
            'type': 'bert-english',
            'max_seq_length': 256,
            'dimensions': 384,
            'performance': 58.80,
            'speed': 14_200,
            'size': 80,
        },
        'paraphrase-MiniLM-L3-v2': {
            'type': 'bert-english',
            'max_seq_length': 128,
            'dimensions': 384,
            'performance': 50.74,
            'speed': 19_000,
            'size': 61,
        },
        # GloVe Model.
        'average_word_embeddings_glove.6B.300d': {
            'type': 'glove',
            'max_seq_length': -1,
            'dimensions': 300,
            'performance': 36.25,
            'speed': 34_000,
            'size': 420,
        },
        # Multilingual Models (50+ languages).
        'paraphrase-multilingual-mpnet-base-v2': {
            'type': 'bert-multilingual',
            'max_seq_length': 128,
            'dimensions': 768,
            'performance': 53.75,
            'speed': 2_500,
            'size': 970,
        },
        'paraphrase-multilingual-MiniLM-L12-v2': {
            'type': 'bert-multilingual',
            'max_seq_length': 128,
            'dimensions': 384,
            'performance': 51.72,
            'speed': 7_500,
            'size': 420,
        }
    }

    def __init__(self, model_name=None, show_progress=False):
        """
        Load or Download the BERT model we are going to use to model the words
        and documents. By default, it loads the fastest model.

        Args:
            model_name: A String with the name of model.
            show_progress: Bool representing whether we show the progress of
                the function or not.
        """
        # Check if a Model was provided, and if we support it.
        if not model_name:
            # Use the fastest model by default.
            model_name = 'average_word_embeddings_glove.6B.300d'
        if model_name not in self.models_dict:
            raise NameError("BertCord19() doesn't support the requested model.")

        # Save Model's name.
        self.model_name = model_name

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

    def model_type(self):
        """
        Give the type of model that was loaded. It can be either Bert or Glove.

        Returns: A string with name of model the class is using.
        """
        model_dict = self.models_dict[self.model_name]
        if 'bert' in model_dict['type']:
            return 'bert'
        else:
            return 'glove'

    def word_vector(self, word):
        """
        Transform a word into a vector.

        Args:
            word: a string of one token.

        Returns:
            The vector of the word.
        """
        result = self.model.encode(word)
        return result

    def document_vector(self, doc_text):
        """
        Transform the text of a document into a vector.

        Args:
            doc_text: A string containing the text of the document.

        Returns:
            The vector of the document.
        """
        result = self.model.encode(doc_text)
        return result


def load_all_models():
    """
    Load or Download all the supported Bert Models to test they work properly.
    """
    # Get The Name of the models
    supported_models = list(BertCord19.models_dict)

    # Record the Runtime of the Program
    new_stopwatch = TimeKeeper()

    for model_name in supported_models:
        print("\n-------------------------------------------------------")
        print(f"Loading Model <{model_name}>:")

        print("\nCreating Bert Model...")
        new_model = BertCord19(model_name=model_name, show_progress=True)
        print("Done.")
        print(f"[{new_stopwatch.formatted_runtime()}]")

        print(f"\nThe Model Type: {new_model.model_type()}")


if __name__ == '__main__':
    # Record the Runtime of the Program
    stopwatch = TimeKeeper()

    # Creating Model.
    print("\nCreating Bert Model...")
    my_model = BertCord19('average_word_embeddings_glove.6B.300d', show_progress=True)
    print("Done.")
    print(f"[{stopwatch.formatted_runtime()}]")

    print(f"\nThe Model Type: {my_model.model_type()}")

    # Test Word Embeddings:
    while True:
        word_input = input("Type a word (q/quit to close): ")
        word_input = word_input.strip().lower()
        if word_input in {'', 'q', 'quit'}:
            break
        word_embed = my_model.word_vector(word_input)
        print(f"\nThe embedding of <{word_input}>:")
        print(word_embed)
        print(f"Type: {type(word_embed)}")

    # Check Similarities between words.
    # print("\nTesting word similarities (To close use [q/quit]):")
    # quit_words = {'q', 'quit'}
    # while True:
    #     word1 = input("\nType the first word: ")
    #     if word1 in quit_words:
    #         break
    #     word2 = input("Type the second word: ")
    #     if word2 in quit_words:
    #         break
    #     sim_words = util.cos_sim(my_model.word_vector(word1), my_model.word_vector(word2))[0][0]
    #     print("The words similarity:")
    #     print(sim_words)

    print("\nDone.")
    print(f"[{stopwatch.formatted_runtime()}]\n")
