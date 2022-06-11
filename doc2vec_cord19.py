# Gelin Eguinosa Rosique

from gensim.models import doc2vec
from gensim.utils import simple_preprocess

from random_sample import RandomSample
from iterable_tokenizer import IterableTokenizer
from time_keeper import TimeKeeper


class Doc2VecCord19:
    """
    Create a Topic Model of the CORD-19 dataset using Doc2Vec.
    """
    # Class Data Locations.
    data_folder = 'project_data'
    doc2vec_model_file = 'doc2vec_cord19_model'
    doc2vec_index = 'doc2vec_index.json'

    def __init__(self, vector_size=300, paper_type='all', corpus_size=-1,
                 use_saved=False, show_progress=False):
        """
        Create a Doc2Vec and Word2Vec model using the CORD-19 dataset.

        Args:
            paper_type: The type of Papers we want to use for the Sample (small,
                medium or big).
            corpus_size: An int with size of the sample. The default value '-1'
                represents all the papers available with the specified paper
                type.
            use_saved: A Bool indicating if we are loading the sample from a
                file.
            show_progress: A Bool representing whether we show the progress of
                the function or not.
        """
        # Load CORD-19 documents.
        self.doc2vec_corpus = RandomSample(paper_type=paper_type,
                                           sample_size=corpus_size,
                                           show_progress=show_progress)
        # Tokenize the documents and Tag them.
        train_corpus = IterableTokenizer(self.doc2vec_corpus.docs_full_texts)
        # train_corpus = _TaggedCorpus(self.doc2vec_corpus.docs_titles_abstracts)

        # Build & Train the Model.
        self.doc2vec_model = doc2vec.Doc2Vec(vector_size=vector_size,
                                             min_count=2,
                                             epochs=40)
        self.doc2vec_model.build_vocab(train_corpus)
        self.doc2vec_model.train(train_corpus,
                                 total_examples=self.doc2vec_model.corpus_count,
                                 epochs=self.doc2vec_model.epochs)
        # Get the Word2Vec model.
        self.word2vec_model = self.doc2vec_model.wv


if __name__ == '__main__':
    # Record the Runtime of the Program
    stopwatch = TimeKeeper()

    # Test the class.
    print("\nCreating Doc2Vec model...")
    test_size = 500
    doc_model = Doc2VecCord19(vector_size=50, paper_type='medium',
                              corpus_size=test_size, show_progress=True)
    print("Done.")
    print(f"[{stopwatch.formatted_runtime()}]")

    # Test Word Vector.
    print("\nWord vector of 'the':")
    print(doc_model.word2vec_model['the'])

    print("\nShow Part of the vocabulary:")
    for index, word in enumerate(doc_model.word2vec_model.index_to_key):
        if index == 10:
            break
        print(f"word #{index}/{len(doc_model.word2vec_model.index_to_key)} is {word}")

    test_word = 'the'
    print(f"\nShow most similar words to '{test_word}':")
    similar_words = doc_model.word2vec_model.most_similar('patient', topn=15)
    for word in similar_words:
        print(f" -> {word}")

    print("\nDone.")
    print(f"[{stopwatch.formatted_runtime()}]\n")
