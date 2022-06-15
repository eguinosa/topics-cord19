# Gelin Eguinosa Rosique

from os import mkdir
from os.path import isdir, isfile, join
from gensim.models import doc2vec

from corpus_cord19 import CorpusCord19
from papers import Papers
from random_sample import RandomSample
from document_model import DocumentModel
from iterable_tokenizer import IterableTokenizer
from doc_tokenizers import doc_tokenizer
from extra_funcs import big_number
from time_keeper import TimeKeeper


class Doc2VecCord19(DocumentModel):
    """
    Create a Topic Model of the CORD-19 dataset using Doc2Vec.
    """
    # Class Data Locations.
    data_folder = 'project_data'
    model_folder = 'doc2vec_models'
    doc2vec_model_file = 'doc2vec_cord19_model'
    doc2vec_model_prefix = 'doc2vec_cord19_model_'
    word2vec_model_file = 'word2vec_model'
    word2vec_model_prefix = 'word2vec_model_'

    def __init__(self, corpus: CorpusCord19, vector_dims=100, show_progress=False,
                 _use_saved=False, _saved_id=None):
        """
        Create a Doc2Vec and Word2Vec model using the CORD-19 dataset.

        Args:
            corpus: The Cord19 Corpus we are going to use to create the Doc2Vec
                model.
            vector_dims: The dimensions for the embeddings of the documents and
                words in the corpus.
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
        model_folder_path = join(self.data_folder, self.model_folder)
        if not isdir(model_folder_path):
            mkdir(model_folder_path)

        if _use_saved:
            # Get the Doc2Vec file we are loading.
            if _saved_id:
                doc2vec_model_name = self.doc2vec_model_prefix + _saved_id
                doc2vec_model_path = join(model_folder_path, doc2vec_model_name)
            else:
                doc2vec_model_path = join(model_folder_path, self.doc2vec_model_file)
            # Check the file exists.
            if not isfile(doc2vec_model_path):
                raise NameError("The Doc2Vec model file doesn't exist.")

            # Load Model.
            self.doc2vec_model = doc2vec.Doc2Vec.load(doc2vec_model_path)
            self.word2vec_model = self.doc2vec_model.wv
        else:
            # Get the CORD-19 documents.
            self.corpus = corpus

            # Create an Iterable with the documents tagged.
            train_corpus = IterableTokenizer(self.corpus.all_papers_content,
                                             tagged_tokens=True)

            # Create and Build the Model.
            if show_progress:
                print("Building Doc2Vec model...")
            self.doc2vec_model = doc2vec.Doc2Vec(vector_size=vector_dims,
                                                 min_count=2,
                                                 epochs=40)
            self.doc2vec_model.build_vocab(train_corpus)

            # Train the Model.
            if show_progress:
                print("Training Doc2Vec model...")
            self.doc2vec_model.train(train_corpus,
                                     total_examples=self.doc2vec_model.corpus_count,
                                     epochs=self.doc2vec_model.epochs)
            # Get the Word2Vec model.
            self.word2vec_model = self.doc2vec_model.wv

            # Save Doc2Vec & Word2Vec Model.
            doc2vec_model_path = join(model_folder_path, self.doc2vec_model_file)
            word2vec_model_path = join(model_folder_path, self.word2vec_model_file)
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
            word: a string of one token.

        Returns:
            The vector of the word.
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
            doc_text: A string containing the text of the document.

        Returns:
            The vector of the document.
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
        # Create files' paths.
        model_folder_path = join(self.data_folder, self.model_folder)
        doc2vec_model_name = self.doc2vec_model_prefix + model_id
        word2vec_model_name = self.word2vec_model_prefix + model_id
        doc2vec_model_path = join(model_folder_path, doc2vec_model_name)
        word2vec_model_path = join(model_folder_path, word2vec_model_name)

        # Save Doc2Vec & Word2Vec Model.
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
        # Create Model Folder if it doesn't exist.
        model_folder_path = join(cls.data_folder, cls.model_folder)
        if not isdir(model_folder_path):
            mkdir(model_folder_path)

        # Create file path.
        if model_id:
            doc2vec_model_name = cls.doc2vec_model_prefix + model_id
            doc2vec_model_path = join(model_folder_path, doc2vec_model_name)
        else:
            doc2vec_model_path = join(model_folder_path, cls.doc2vec_model_file)

        # Check if the file exists.
        result = isfile(doc2vec_model_path)
        return result


if __name__ == '__main__':
    # Record the Runtime of the Program
    stopwatch = TimeKeeper()

    # Test the class.
    test_size = 500
    print(f"\nLoading Random Sample of {big_number(test_size)} documents...")
    model_corpus = RandomSample('medium', test_size, show_progress=True)
    print("Done.")
    print(f"[{stopwatch.formatted_runtime()}]")

    # Create Doc2Vec model.
    print("\nCreating Doc2Vec model...")
    doc_model = Doc2VecCord19(corpus=model_corpus, vector_dims=300,
                              show_progress=True)
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
        print(doc_model.word_vector(input_word))

    # print("\nTesting loading Doc2Vec model...")
    # print("Loading model...")
    # next_doc2vec = Doc2VecCord19.load(show_progress=True)
    # print("Done.")
    # print(f"[{stopwatch.formatted_runtime()}]")
    #
    # print("\nShow Loaded Doc2Vec Model vocabulary:")
    # for index, doc_word in enumerate(next_doc2vec.word2vec_model.index_to_key):
    #     if index == 20:
    #         break
    #     print(f"word #{index}/{len(doc_model.word2vec_model.index_to_key)} is {doc_word}")
    #
    # test_word = 'virus'
    # print(f"\nShow most similar words to '{test_word}' in loaded Model:")
    # similar_words = next_doc2vec.word2vec_model.most_similar('patient', topn=15)
    # for sim_word in similar_words:
    #     print(f" -> {sim_word}")

    print("\nDone.")
    print(f"[{stopwatch.formatted_runtime()}]\n")

    # **************************************************************
    # <--- To Create Doc2Vec model of the entire CORD-19 corpus --->
    # (Done - 9 hours for 138,967 documents)
    # **************************************************************

    # # Load Corpus.
    # print(f"\nLoading CORD-19 corpus...")
    # model_corpus = Papers(show_progress=True)
    # print("Done.")
    # print(f"[{stopwatch.formatted_runtime()}]")

    # # Check the amount of documents loaded.
    # num_papers = len(model_corpus.papers_index)
    # print(f"\nThe current CORD-19 dataset has {big_number(num_papers)} documents.")

    # # Create Doc2Vec model.
    # print("\nCreating Doc2Vec model...")
    # doc_model = Doc2VecCord19(corpus=model_corpus, vector_dims=300,
    #                           show_progress=True)
    # print("Done.")
    # print(f"[{stopwatch.formatted_runtime()}]")

    # # Save CORD-19 Corpus Word2Vec Model.
    # print("\nSaving the Doc2Vec Model of the CORD-19 Dataset...")
    # doc_model.save_model('cord19_dataset')
    # print("Done.")
    # print(f"[{stopwatch.formatted_runtime()}]")
