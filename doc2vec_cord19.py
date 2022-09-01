# Gelin Eguinosa Rosique

from os import mkdir, listdir
from os.path import isdir, isfile, join
from gensim.models import doc2vec

from corpus_cord19 import CorpusCord19
from papers_cord19 import PapersCord19
from random_sample import RandomSample
from document_model import DocumentModel
from iterable_tokenizer import IterableTokenizer
from doc_tokenizers import doc_tokenizer
from extra_funcs import big_number, progress_msg
from time_keeper import TimeKeeper


class Doc2VecCord19(DocumentModel):
    """
    Create a Topic Model of the CORD-19 dataset using Doc2Vec.
    """
    # Class Data Locations.
    data_folder = 'project_data'
    class_data_folder = 'doc2vec_models'
    model_folder_prefix = 'model_'
    doc2vec_model_prefix = 'doc2vec_'
    word2vec_model_prefix = 'word2vec_'
    default_model_id = 'default'

    def __init__(self, corpus: CorpusCord19 = None, vector_dims=100,
                 use_title_abstract=False, show_progress=False,
                 _use_saved=False, _saved_id=None):
        """
        Create a Doc2Vec and Word2Vec model using the CORD-19 dataset. If no
        corpus is provided, use the Papers() class containing all the papers in
        the CORD-19 dataset.

        Args:
            corpus: The Cord19 Corpus we are going to use to create the Doc2Vec
                model.
            vector_dims: The dimensions for the embeddings of the documents and
                words in the corpus.
            use_title_abstract: Bool indicating whether we are using the title
                and abstract or the full content of the documents to create the
                Doc2Vec model.
            show_progress: A Bool representing whether we show the progress of
                the function or not.
            _use_saved: A Bool indicating if we are loading the sample from a
                file.
            _saved_id: The ID of the model we want to load, if the model was
                saved with an ID.
        """
        # Create Project Folder if it doesn't exist.
        if not isdir(self.data_folder):
            mkdir(self.data_folder)
        # Create Model Folder if it doesn't exist.
        class_folder_path = join(self.data_folder, self.class_data_folder)
        if not isdir(class_folder_path):
            mkdir(class_folder_path)

        if _use_saved:
            # Create filename of the model we are loading.
            if _saved_id:
                model_folder_name = self.model_folder_prefix + _saved_id
                doc2vec_filename = self.doc2vec_model_prefix + _saved_id
            else:
                model_folder_name = self.model_folder_prefix + self.default_model_id
                doc2vec_filename = self.doc2vec_model_prefix + self.default_model_id

            # Check the model is properly saved.
            model_folder_path = join(class_folder_path, model_folder_name)
            doc2vec_model_path = join(model_folder_path, doc2vec_filename)
            if not isdir(model_folder_path):
                raise NameError("The folder of the Doc2Vec model does not exist.")
            if not isfile(doc2vec_model_path):
                raise NameError("The Doc2Vec model file doesn't exist.")

            # Load Model.
            self.doc2vec_model = doc2vec.Doc2Vec.load(doc2vec_model_path)
            self.word2vec_model = self.doc2vec_model.wv
        else:
            # Check if we have a corpus and get the CORD-19 documents.
            if corpus:
                self.corpus = corpus
            else:
                if show_progress:
                    progress_msg("Creating default CORD-19 corpus...")
                self.corpus = PapersCord19(show_progress=show_progress)

            # Create an Iterable with the documents tagged.
            if use_title_abstract:
                train_corpus = IterableTokenizer(self.corpus.all_papers_title_abstract,
                                                 tagged_tokens=True)
            else:
                train_corpus = IterableTokenizer(self.corpus.all_papers_content,
                                                 tagged_tokens=True)

            # Create and Build the Model.
            if show_progress:
                progress_msg("Building Doc2Vec model...")
            self.doc2vec_model = doc2vec.Doc2Vec(vector_size=vector_dims,
                                                 min_count=2,
                                                 epochs=40)
            self.doc2vec_model.build_vocab(train_corpus)

            # Train the Model.
            if show_progress:
                progress_msg("Training Doc2Vec model...")
            self.doc2vec_model.train(train_corpus,
                                     total_examples=self.doc2vec_model.corpus_count,
                                     epochs=self.doc2vec_model.epochs)
            # Get the Word2Vec model.
            self.word2vec_model = self.doc2vec_model.wv

            # Creating Model File & Folder Name.
            model_folder_name = self.model_folder_prefix + self.default_model_id
            doc2vec_filename = self.doc2vec_model_prefix + self.default_model_id
            word2vec_filename = self.word2vec_model_prefix + self.default_model_id
            # Create Model Folder.
            model_folder_path = join(class_folder_path, model_folder_name)
            if not isdir(model_folder_path):
                mkdir(model_folder_path)
            # Save Doc2Vec & Word2Vec Model.
            doc2vec_model_path = join(model_folder_path, doc2vec_filename)
            word2vec_model_path = join(model_folder_path, word2vec_filename)
            self.doc2vec_model.save(doc2vec_model_path)
            self.word2vec_model.save(word2vec_model_path)

    def model_type(self):
        """
        Give the type of Document Model this class is.

        Returns: A string with the name of the model the class is using.
        """
        return 'doc2vec'

    def word_vector(self, word):
        """
        Transform a word into a vector.

        Args:
            word: String of the token word.

        Returns:
            Numpy.ndarray with the vector of the word.
        """
        # Check if the word exists in the dictionary of the model.
        if word not in self.word2vec_model:
            return [0]
        # Use the Word2Vec Model.
        result = self.word2vec_model[word]
        return result

    def document_vector(self, doc_text):
        """
        Transform the text of a document into a vector.

        Args:
            doc_text: String containing the text of the document.

        Returns:
            Numpy.ndarray with the vector of the document.
        """
        # Tokenize the document text.
        doc_tokens = doc_tokenizer(doc_text)
        # Get vector using the trained Doc2Vec model.
        doc_vector = self.doc2vec_model.infer_vector(doc_tokens)
        return doc_vector

    def save_model(self, model_id):
        """
        Save the Doc2Vec model in a different file with a Name ID, to use it
        later.

        Args:
            model_id: The Identifier add to the model filename.
        """
        # Creating Model File & Folder Name.
        model_folder_name = self.model_folder_prefix + model_id
        doc2vec_filename = self.doc2vec_model_prefix + model_id
        word2vec_filename = self.word2vec_model_prefix + model_id

        # Create Model Folder.
        model_folder_path = join(self.data_folder, self.class_data_folder, model_folder_name)
        if not isdir(model_folder_path):
            mkdir(model_folder_path)

        # Save Doc2Vec & Word2Vec Model.
        doc2vec_model_path = join(model_folder_path, doc2vec_filename)
        word2vec_model_path = join(model_folder_path, word2vec_filename)
        self.doc2vec_model.save(doc2vec_model_path)
        self.word2vec_model.save(word2vec_model_path)

    @classmethod
    def load(cls, model_id=None, show_progress=False):
        """
        Load a previously created Doc2Vec model.
        - If no 'model_id' is provided, it loads the last Doc2Vec model created.

        Args:
            model_id: A String with the ID the model we want to load was saved
                with.
            show_progress: A Bool representing whether we show the progress of
                the function or not.

        Returns:
            A Doc2VecCord19().
        """
        # Load Doc2Vec Model.
        new_doc2vec_model = cls(corpus=None, _use_saved=True, _saved_id=model_id,
                                show_progress=show_progress)
        return new_doc2vec_model

    @classmethod
    def saved_models(cls):
        """
        Create a list with the IDs of all the Doc2Vec models saved.

        Returns: List[str] with the IDs of the models.
        """
        # Check if the Project Folder exists.
        if not isdir(cls.data_folder):
            return []
        # Check if the Class Folder exists.
        class_folder_path = join(cls.data_folder, cls.class_data_folder)
        if not isdir(class_folder_path):
            return []

        # Find all the available Model Folders.
        models_ids = []
        for element_name in listdir(class_folder_path):
            element_path = join(class_folder_path, element_name)
            if not isdir(element_path):
                continue
            if not element_name.startswith(cls.model_folder_prefix):
                continue
            prefix_len = len(cls.model_folder_prefix)
            new_model_id = element_name[prefix_len:]
            models_ids.append(new_model_id)

        # The IDs of the saved Doc2Vec Models.
        return models_ids

    @classmethod
    def model_saved(cls, model_id=None):
        """
        Check if there is a Doc2Vec model saved.
        - If no 'model_id' is provided, checks if the last used Doc2Vec model
        was saved.

        Args:
            model_id: A String with the ID of the model we want to check.

        Returns:
            A Bool showing if there is a model saved or not.
        """
        # Create Project Folder if it doesn't exist.
        if not isdir(cls.data_folder):
            mkdir(cls.data_folder)
        # Create Class Folder if it doesn't exist.
        class_folder_path = join(cls.data_folder, cls.class_data_folder)
        if not isdir(class_folder_path):
            mkdir(class_folder_path)

        # Create the Model's Folder & File name.
        if model_id:
            model_folder_name = cls.model_folder_prefix + model_id
            doc2vec_filename = cls.doc2vec_model_prefix + model_id
        else:
            model_folder_name = cls.model_folder_prefix + cls.default_model_id
            doc2vec_filename = cls.doc2vec_model_prefix + cls.default_model_id

        # Check if the Model's Folder exists.
        model_folder_path = join(class_folder_path, model_folder_name)
        if not isdir(model_folder_path):
            return False
        # Check if the Model's File exists.
        doc2vec_model_path = join(model_folder_path, doc2vec_filename)
        if not isdir(doc2vec_model_path):
            return False

        # All Good.
        return True


if __name__ == '__main__':
    # Record the Runtime of the Program
    stopwatch = TimeKeeper()

    # Test the class.
    test_size = 500
    print(f"\nLoading Random Sample of {big_number(test_size)} documents...")
    model_corpus = RandomSample(paper_type='medium', sample_size=test_size, show_progress=True)
    print("Done.")
    print(f"[{stopwatch.formatted_runtime()}]")

    # Create Doc2Vec model.
    print("\nCreating Doc2Vec model...")
    doc_model = Doc2VecCord19(corpus=model_corpus, vector_dims=300,
                              use_title_abstract=True, show_progress=True)
    # doc_model = Doc2VecCord19.load(show_progress=True)
    print("Done.")
    print(f"[{stopwatch.formatted_runtime()}]")

    # Test Word Vector.
    print("\nWord vector of 'the':")
    print(doc_model.word2vec_model['the'])

    print("\nShow Part of the vocabulary:")
    for index, doc_word in enumerate(doc_model.word2vec_model.index_to_key):
        if index == 20:
            break
        print(f"word #{index}/{len(doc_model.word2vec_model.index_to_key)} is {doc_word}")

    test_word = 'virus'
    print(f"\nShow most similar words to '{test_word}':")
    similar_words = doc_model.word2vec_model.most_similar('patient', topn=15)
    for sim_word in similar_words:
        print(f" -> {sim_word}")

    # Get the embedding of input words.
    while True:
        input_word = input("\nFrom which word do you want to get the embedding?\n ")
        if not input_word or input_word in {'q', 'quit'}:
            break
        print(f"Word vector of {input_word}:")
        the_word_vector = doc_model.word_vector(input_word)
        print(the_word_vector)
        print(f"Type: {type(the_word_vector)}")

    # Save Model.
    new_input = input("Do you want to save this model? (yes/[no]) ")
    new_input = new_input.strip().lower()
    if new_input in {'y', 'yes'}:
        new_name = input("Type the name of the model: ")
        new_name = new_name.strip().lower()
        doc_model.save_model(model_id=new_name)

    # # --- Test Loading Saved Models ---
    # print("\nSaved Models:")
    # the_saved_models = Doc2VecCord19.saved_models()
    # print(the_saved_models)
    #
    # the_model_name = 'default'
    # print(f"\nLoading Model <{the_model_name}>...")
    # next_doc2vec = Doc2VecCord19.load(model_id=the_model_name, show_progress=True)
    # print("Done.")
    # print(f"[{stopwatch.formatted_runtime()}]")
    #
    # print("\nShow Loaded Doc2Vec Model vocabulary:")
    # for index, doc_word in enumerate(next_doc2vec.word2vec_model.index_to_key):
    #     if index == 20:
    #         break
    #     print(f"word #{index}/{len(next_doc2vec.word2vec_model.index_to_key)} is {doc_word}")
    #
    # test_word = 'virus'
    # print(f"\nShow most similar words to '{test_word}' in loaded Model:")
    # similar_words = next_doc2vec.word2vec_model.most_similar('patient', topn=15)
    # for sim_word in similar_words:
    #     print(f" -> {sim_word}")

    print("\nDone.")
    print(f"[{stopwatch.formatted_runtime()}]\n")
