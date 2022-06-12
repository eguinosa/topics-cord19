# Gelin Eguinosa Rosique

from abc import ABC, abstractmethod


class DocumentModel(ABC):
    """
    Abstract class with the methods that the Document Models in the project
    need to implement.
    """

    @abstractmethod
    def model_type(self):
        """
        Get the type of the Document Model. It can be ['doc2vec', 'bert',
        'specter', etc...].

        Returns: A string with name of model the class is using.
        """
        pass

    @abstractmethod
    def word_vector(self, word):
        """
        Get the vector embedding for the given 'word' string.
        """
        pass

    @abstractmethod
    def document_vector(self, document):
        """
        Get the embedding for the text of the provided 'document' string.
        """
        pass

    def words_vectors(self, words):
        """
        Get the embeddings for the provided list of words.

        Args:
            words: List of strings containing the words.

        Returns:
            A list of word vectors.
        """
        # Use self.word_vector()
        vectors_list = []
        for word in words:
            new_vector = self.word_vector(word)
            vectors_list.append(new_vector)

        # List containing all the word vectors.
        return vectors_list

    def documents_vectors(self, documents):
        """
        Get the vector embeddings for each of the documents in the 'documents'
        list.

        Args:
            documents: A list of strings with the content of the documents.

        Returns:
             A list with the vector of the documents.
        """
        # Use self.document_vector()
        vectors_list = []
        for doc_text in documents:
            new_vector = self.document_vector(doc_text)
            vectors_list.append(new_vector)

        # List of the doc vectors.
        return vectors_list
