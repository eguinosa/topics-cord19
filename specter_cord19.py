# Gelin Eguinosa Rosique

# import json
from os import mkdir
from os.path import isdir, join
from transformers import AutoTokenizer, AutoModel

from document_model import DocumentModel
from time_keeper import TimeKeeper
from extra_funcs import progress_bar, progress_msg


class SpecterCord19(DocumentModel):
    """
    Load and Manage the Specter Model to get the vector representations of the
    word and documents in the CORD-19 corpus.
    """
    # Data Locations.
    project_models_folder = 'doc_models'
    specter_folder = 'specter_model_tokenizer'
    tokenizer_folder = 'specter_tokenizer'
    model_folder = 'specter_model'

    def __init__(self, show_progress=False):
        """
        Load the tokenizer and model of specter. Save the model if it wasn't
        available locally.

        Args:
            show_progress: Bool representing whether we show the progress of
                the function or not.
        """
        # Check the project and class folder exist.
        if not isdir(self.project_models_folder):
            mkdir(self.project_models_folder)
        specter_folder_path = join(self.project_models_folder, self.specter_folder)
        if not isdir(specter_folder_path):
            mkdir(specter_folder_path)

        # Load or Download tokenizer.
        tokenizer_path = join(specter_folder_path, self.tokenizer_folder)
        if not isdir(tokenizer_path):
            if show_progress:
                progress_msg("Downloading Tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
            self.tokenizer.save_pretrained(tokenizer_path)
        else:
            if show_progress:
                progress_msg("Loading Tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Load or Download model.
        model_path = join(specter_folder_path, self.model_folder)
        if not isdir(model_path):
            if show_progress:
                progress_msg("Downloading Model...")
            self.model = AutoModel.from_pretrained('allenai/specter')
            self.model.save_pretrained(model_path)
        else:
            if show_progress:
                progress_msg("Loading Model...")
            self.model = AutoModel.from_pretrained(model_path)

    def model_type(self):
        """
        Give the type of Document Model this class is: Specter

        Returns: A string with the model type.
        """
        return 'specter'

    def word_vector(self, word):
        """
        Give the embedding for the given 'word'.

        Args:
            word: A string with the word we want to encode.

        Returns:
            A Numpy Array containing the embedding of the word.
        """
        # Tokenize and Encode word.
        word_inputs = self.tokenizer([word], padding=True, truncation=True,
                                     return_tensors="pt", max_length=512)
        input_embeds = self.model(**word_inputs)
        word_embed = input_embeds.last_hidden_state[:, 0, :][0]

        # Transform the tensor embedding into a numpy array.
        numpy_embed = word_embed.detach().numpy()
        return numpy_embed

    def document_vector(self, title_abstract):
        """
        Create the embedding of a paper providing the title and abstract of the
        document.

        Args:
            title_abstract: A string containing the title and abstract of the
                paper.

        Returns:
           A Numpy Array with the embedding of the document.
        """
        # Tokenize and Encode document.
        doc_inputs = self.tokenizer([title_abstract], padding=True, truncation=True,
                                    return_tensors="pt", max_length=512)
        input_embeds = self.model(**doc_inputs)
        doc_embed = input_embeds.last_hidden_state[:, 0, :][0]

        # Transform the tensor embedding into a numpy array.
        numpy_embed = doc_embed.detach().numpy()
        return numpy_embed

    def create_vocab_dict(self, vocabulary, show_progress=False):
        """
        Create a dictionary containing the Specter embedding of all the words
        contained in 'vocabulary'.

        Args:
            vocabulary: List of strings containing the words in the vocabulary.
            show_progress: Bool representing whether we show the progress of
                the function or not.

        Returns:
            A Dictionary with the String of the words as keys, and a List[float]
                with their embeddings as values.
        """
        # Progress variables.
        count = 0
        total = len(vocabulary)

        # Create dictionary containing the embeddings.
        vocab_embeds = {}
        for vocab_word in vocabulary:
            # Get numpy embedding and transform it to a List[float]
            word_embed = self.word_vector(vocab_word)
            embed_list = word_embed.tolist()
            vocab_embeds[vocab_word] = embed_list

            # Progress.
            if show_progress:
                count += 1
                progress_bar(count, total)

        # Dictionary with the vocabulary's words and their embeddings.
        return vocab_embeds


if __name__ == '__main__':
    # Record the Runtime of the Program
    stopwatch = TimeKeeper()

    print("\nCreating Specter Model...")
    specter_model = SpecterCord19(show_progress=True)
    print("Done.")
    print(f"[{stopwatch.formatted_runtime()}]")

    # Get the embeddings from input words.
    while True:
        input_word = input("\nFrom which word do you want to get the embedding?\n ")
        if not input_word or input_word in {'q', 'quit'}:
            break
        print(f"Word vector of {input_word}:")
        word_embedding = specter_model.word_vector(input_word)
        print(type(word_embedding))
        print(word_embedding)

    # # Test getting embeddings for a list of words.
    # the_vocabulary = ['jelly', 'peanut', 'food', 'beach', 'party', 'feast',
    #                   'adventure', 'funny', 'sadness', 'country', 'summer']
    # print("\nCreating embeddings for the vocabulary:")
    # print(the_vocabulary)
    # the_vocab_embeds = specter_model.create_vocab_dict(the_vocabulary, show_progress=True)
    # print("Done.")
    # print(f"[{stopwatch.formatted_runtime()}]")
    #
    # the_embeds_folder = 'temp_vocab_embeds_dict.json'
    # print("\nSaving vocabulary embeddings in:")
    # print(the_embeds_folder)
    # with open(the_embeds_folder, 'w') as f:
    #     json.dump(the_vocab_embeds, f)
    # print("Done.")
    # print(f"[{stopwatch.formatted_runtime()}]")
