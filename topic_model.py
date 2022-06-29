# Gelin Eguinosa Rosique

import json
import umap
import hdbscan
import numpy as np
from os import mkdir
from os.path import isdir, isfile, join
from numpy.linalg import norm
from sentence_transformers import util

from corpus_cord19 import CorpusCord19
from papers import Papers
from document_model import DocumentModel
from random_sample import RandomSample
from bert_cord19 import BertCord19
from specter_manager import SpecterManager
from doc2vec_cord19 import Doc2VecCord19
from doc_tokenizers import doc_tokenizer
from extra_funcs import progress_bar, big_number
from time_keeper import TimeKeeper


class TopicModel:
    """
    Find the topics in the CORD-19 corpus using the method described in Top2Vec.
    """
    # Class Data Locations.
    data_folder = 'project_data'
    topic_models_folder = 'topic_models'
    topic_model_prefix = 'topics_'  # + doc model + docs number + id
    model_index_file = 'topic_model_index.json'
    word_embeds_file = 'topic_model_word_embeds.json'
    doc_embeds_file = 'topic_model_doc_embeds.json'
    topic_embeds_file = 'topic_model_topic_embeds.json'

    # The Doc Models available to use.
    supported_doc_models = ['doc2vec', 'glove', 'bert', 'specter']

    def __init__(self, corpus: CorpusCord19 = None, doc_model: DocumentModel = None,
                 only_title_abstract=False, show_progress=False,
                 _used_saved=False, _model_id=None):
        """
        Find the topics in the provided 'corpus' using 'doc_model' to get the
        embedding of the Documents and Words in the CORD-19 corpus selected.
            - If '_used_saved' is True, loads a previously used and saved model
              depending on the value of '_saved_id'.
            - If no ID is provided in 'saved_id', load the last used model.
            - If no 'corpus' is provided, use Papers().
            - If no 'doc_model' is provided, use BertCord19().

        Args:
            corpus: A Cord-19 Corpus class with a selection of papers.
            doc_model: A Document Model class used to get the embeddings of the
                words and documents in the corpus.
            only_title_abstract: A Bool showing if we are going to use only the
                Titles & Abstracts of the papers, or all their content to create
                the vocabulary and the embeddings.
            show_progress: Bool representing whether we show the progress of
                the function or not.
            _used_saved: A Bool to know if we need to load the Topic Model from
                a file or recalculate it.
            _model_id: A string with the ID of a previously saved Topic Model.
        """
        if _used_saved:
            # -- Load Topic Model ---
            # Check if the data folders exist.
            if not isdir(self.data_folder):
                raise FileNotFoundError("There is no project data available.")
            topic_models_path = join(self.data_folder, self.topic_models_folder)
            if not isdir(topic_models_path):
                raise FileNotFoundError("There is no Topic Model saved.")

            # Create folder path for this topic model.
            if _model_id:
                model_folder_id = self.topic_model_prefix + _model_id
            else:
                model_folder_id = self.topic_model_prefix + 'default'
            model_folder_path = join(topic_models_path, model_folder_id)

            # Load Index Dictionary for basic attributes.
            if show_progress:
                print("Loading Topic Model basic attributes...")
            model_index_path = join(model_folder_path, self.model_index_file)
            if not isfile(model_index_path):
                raise FileNotFoundError("There is no Attributes Index available.")
            with open(model_index_path, 'r') as f:
                topic_model_index = json.load(f)
            # Get the Attributes.
            self.model_type = topic_model_index['model_type']
            self.use_title_abstract = topic_model_index['use_title_abstract']
            self.num_topics = topic_model_index['num_topics']
            self.corpus_ids = topic_model_index['corpus_ids']
            self.topic_docs = topic_model_index['topic_docs']
            self.topic_words = topic_model_index['topic_words']

            # Load Word Embeddings Dictionary.
            if show_progress:
                print("Loading Topic Model's word embeddings...")
            word_index_path = join(model_folder_path, self.word_embeds_file)
            if not isfile(word_index_path):
                raise FileNotFoundError("There is no Word Embeddings available.")
            with open(word_index_path, 'r') as f:
                word_embeds_index = json.load(f)
            # Progress Variables.
            count = 0
            total = len(word_embeds_index)
            # Transform the Word Embeddings to Numpy.ndarray
            self.word_embeds = {}
            for word_key, word_embed in word_embeds_index.items():
                self.word_embeds[word_key] = np.array(word_embed)
                if show_progress:
                    count += 1
                    progress_bar(count, total)

            # Load Doc Embeddings Dictionary.
            if show_progress:
                print("Loading Topic Model's doc embeddings...")
            doc_index_path = join(model_folder_path, self.doc_embeds_file)
            if not isfile(doc_index_path):
                raise FileNotFoundError("There is no Doc Embeddings available.")
            with open(doc_index_path, 'r') as f:
                doc_embeds_index = json.load(f)
            # Progress Variables.
            count = 0
            total = len(doc_embeds_index)
            # Transform Doc Embeddings to Numpy.ndarray
            self.doc_embeds = {}
            for doc_key, doc_embed in doc_embeds_index.items():
                self.doc_embeds[doc_key] = np.array(doc_embed)
                if show_progress:
                    count += 1
                    progress_bar(count, total)

            # Load Topic Embeddings Dictionary.
            if show_progress:
                print("Loading Topic's Embeddings...")
            topic_index_path = join(model_folder_path, self.topic_embeds_file)
            if not isfile(topic_index_path):
                raise FileNotFoundError("There is no Topic Embeddings available.")
            with open(topic_index_path, 'r') as f:
                topic_embeds_index = json.load(f)
            # Progress Variables.
            count = 0
            total = len(topic_embeds_index)
            # Transform Topic Embeddings to Numpy.ndarray
            self.topic_embeds = {}
            for topic_key, topic_embed in topic_embeds_index.items():
                self.topic_embeds[topic_key] = np.array(topic_embed)
                if show_progress:
                    count += 1
                    progress_bar(count, total)
        else:
            # -- Create Topic Model --
            # Load CORD-19 Corpus.
            if corpus:
                self.corpus = corpus
            else:
                self.corpus = Papers()
            self.corpus_ids = self.corpus.papers_cord_uids()

            # Load Document Model.
            if doc_model:
                self.doc_model = doc_model
            else:
                self.doc_model = BertCord19()
            self.model_type = self.doc_model.model_type()

            # Save if we are only using the Title and Abstract of the Papers.
            self.use_title_abstract = only_title_abstract

            # Check we are using a supported model.
            if self.model_type not in self.supported_doc_models:
                raise NameError(f"The model type <{self.model_type}> is not supported.")

            # Calculate the embeddings of the words and documents.
            if show_progress:
                print("Creating Word Embeddings...")
            self.word_embeds = self._create_word_embeddings(show_progress=show_progress)
            if show_progress:
                print("Creating Document Embeddings...")
            self.doc_embeds = self._create_docs_embeddings(show_progress=show_progress)
            if show_progress:
                print("Finding Topics...")
            self.topic_embeds = self._find_topics(show_progress=show_progress)
            self.num_topics = len(self.topic_embeds)
            if show_progress:
                print(f"{self.num_topics} topics found.")
            if show_progress:
                print("Organizing documents by topics...")
            self.topic_docs = self._find_doc_topics(show_progress=show_progress)
            if show_progress:
                print("Creating topics vocabulary...")
            self.topic_words = self._find_word_topics(show_progress=show_progress)

            # Create Default values for topics created with a fixed number of
            # desired topics:
            # ----------------------------------------------------------------
            # Bool indicating if we have hierarchically reduced topics.
            self.new_topics = False
            # Create attributes for the Hierarchical Topic Reduction.
            self.new_topic_embeds = None
            self.new_num_topics = None
            self.new_topic_docs = None
            self.new_topic_words = None
            # Dictionary with new topics as keys, and their closest original
            # topics as values.
            self.topics_hierarchy = None

    def generate_new_topics(self, req_num_topics: int, show_progress=False):
        """
        Create a new Hierarchical Topic Model with specified number of topics
        (num_topics). The 'num_topics' need to be at least 2 topics, and be
        smaller than the original number of topics found.

        Args:
            req_num_topics: The desired topic count for the new Topic Model.
            show_progress: Bool representing whether we show the progress of
                the function or not.
        """
        # Check the number of topics requested is valid.
        if 1 < req_num_topics < self.num_topics:
            # Invalid topic size requested. Reset reduced topics variables.
            self.new_topics = False
            self.new_topic_embeds = None
            self.new_num_topics = None
            self.new_topic_docs = None
            self.new_topic_words = None
            self.topics_hierarchy = None
            # Progress.
            if show_progress:
                print("Invalid number of topics requested. No hierarchical topic"
                      " reduction performed.")
            # Exit function.
            return

        # Check if we can use previously calculated New Topics...
        # [...Not Done Yet]

        # Initialize New Topics Variables.
        current_num_topics = self.num_topics
        new_topic_embeds = self.topic_embeds.copy()
        new_topic_sizes = dict([(topic_id, len(self.topic_docs[topic_id]))
                                for topic_id in self.topic_docs.keys()])

        # Progress Variables.
        count = 0
        total = current_num_topics - req_num_topics
        # Perform topic reduction until we get the desired number of topics
        while req_num_topics < current_num_topics:
            # Get the smallest topic.
            new_topics_list = list(new_topic_embeds.keys())
            min_topic_id = min(new_topics_list, key=lambda x: len(self.topic_docs[x]))
            min_embed = new_topic_embeds[min_topic_id]

            # Get the closest topic to the small topic.
            close_topic_id, _ = self._closest_topic(min_embed)
            close_embed = new_topic_embeds[close_topic_id]

            # Merge the embedding of the topics.
            min_size = new_topic_sizes[min_topic_id]
            close_size = new_topic_sizes[close_topic_id]
            total_size = min_size + close_size
            merged_topic_embed = (min_size * min_embed + close_size * close_embed) / total_size

            # Delete Smallest Topic.
            del new_topic_embeds[min_topic_id]
            # Update the Embedding of the closest Topic to the mean of them both.
            new_topic_embeds[close_topic_id] = merged_topic_embed

            # Update Document Count per Topic.
            pass


    def top_topics(self):
        """
        Make a sorted list with the topics organized by the amount of documents
        they represent.

        Returns:
            A list of tuples with the topics' ID and their document count.
        """
        # Form list of topics with their size.
        topic_docs = [(topic_id, len(docs_list))
                      for topic_id, docs_list in self.topic_docs.items()]
        # Sort by size.
        topic_docs.sort(key=lambda count: count[1], reverse=True)
        return topic_docs

    def top_words_topic(self, topic_id, num_words=10):
        """
        Find the top n words for the given topic. If 'num_words' is -1, then
        return all the words belonging to this topic.

        Args:
            topic_id: The topic from which we want the top words.
            num_words: The amount of words from that topic that we want.

        Returns:
            A list of tuples with the words and their similarity to the topic.
        """
        # Check the topic exists.
        if topic_id not in self.topic_words:
            raise NameError("Topic not found.")

        # Get the list of words we are returning.
        if num_words == -1:
            # All words in the topic.
            result = self.topic_words[topic_id]
        else:
            # Check we are not giving more words than what we can.
            word_count = min(num_words, len(self.topic_words[topic_id]))
            # Get the number of words requested.
            result = self.topic_words[topic_id][:word_count]
        # List of tuples with words and similarities.
        return result

    def all_topics_top_words(self, num_words=10):
        """
        Make a list with the top words per topics. If 'num_words' is -1, returns
        all the words belonging to a topic.

        Returns: A list of tuples with the Topic ID and their top_words_topic(),
            the latter containing the top words for the corresponding topic.
        """
        topics_top_words = []
        for topic_id, _ in self.top_topics():
            topic_top_words = self.top_words_topic(topic_id, num_words=num_words)
            topics_top_words.append((topic_id, topic_top_words))
        # Per topic, a list with the words and their similarity to the topic.
        return topics_top_words

    def _create_word_embeddings(self, show_progress=False):
        """
        Create a dictionary with all the words in the corpus and their embeddings.

        Args:
            show_progress: A Bool representing whether we show the progress of
                the function or not.

        Returns:
            Dictionary with the words and embeddings of the corpus' vocabulary.
        """
        # Progress Bar variables.
        count = 0
        total = len(self.corpus_ids)

        # Check if we are only using Title & Abstract or the whole papers' content.
        if self.use_title_abstract:
            content_provider = self.corpus.all_papers_title_abstract()
        else:
            content_provider = self.corpus.all_papers_content()

        # Create the vocabulary using the tokens from the Corpus' Content.
        words_embeddings = {}
        for doc_content in content_provider:
            doc_tokens = doc_tokenizer(doc_content)
            # Add the new words from the document.
            for doc_word in doc_tokens:
                if doc_word not in words_embeddings:
                    word_embed = self.doc_model.word_vector(doc_word)
                    # Ignore words that the model can't encode (Zero Values).
                    if not np.any(word_embed):
                        continue
                    words_embeddings[doc_word] = word_embed
            # Show Progress.
            if show_progress:
                count += 1
                progress_bar(count, total)

        # The dictionary of words with their embeddings.
        return words_embeddings

    def _create_docs_embeddings(self, show_progress=False):
        """
        Calculate the embeddings of the documents using the specified Document
        Model.

        Args:
            show_progress: A Bool representing whether we show the progress of
                the function or not.

        Returns:
            A dictionary containing the cord_uids of the documents in the CORD-19
                sample and their embeddings.
        """
        # Progress Bar variables.
        count = 0
        total = len(self.corpus_ids)

        docs_embeddings = {}
        for cord_uid in self.corpus_ids:
            # Depending on the model, select the content of the paper to encode.
            if self.model_type == 'specter':
                # Using the Specter Manager Now.
                doc_embedding = self.doc_model.document_vector(cord_uid)
            elif self.use_title_abstract or self.model_type == 'bert':
                # BERT has a length limit, use only title and abstract.
                doc_content = self.corpus.paper_title_abstract(cord_uid)
                doc_embedding = self.doc_model.document_vector(doc_content)
            elif self.model_type in {'doc2vec', 'glove'}:
                # No text size or token count restrictions, use all text.
                doc_content = self.corpus.paper_content(cord_uid)
                doc_embedding = self.doc_model.document_vector(doc_content)
            else:
                raise NameError(f"We don't support the Model<{self.model_type}> yet.")

            # Save Document's embedding.
            docs_embeddings[cord_uid] = doc_embedding

            # Progress.
            if show_progress:
                count += 1
                progress_bar(count, total)

        # The dictionary with the Docs IDs and their embeddings.
        return docs_embeddings

    def _find_topics(self, show_progress=False):
        """
        Find the corpus' topics using the embeddings of the documents.

        Args:
            show_progress: A Bool representing whether we show the progress of
                the function or not.
        Returns:
            A dictionary with the topics and their embeddings.
        """
        # Get Document Embeddings.
        doc_embeddings = list(self.doc_embeds.values())

        # Use UMAP to reduce the dimensions of the embeddings.
        if show_progress:
            print("UMAP: Reducing dimensions of the papers...")
        umap_embeddings = umap.UMAP(n_neighbors=15,
                                    n_components=5,
                                    metric='cosine').fit_transform(doc_embeddings)

        # Use HDBSCAN to find the cluster of documents in the vector space.
        if show_progress:
            print("HDBSCAN: Creating topic clusters with the documents...")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=15,
                                    metric='euclidean',
                                    cluster_selection_method='eom')
        cluster_labels = clusterer.fit(umap_embeddings)

        # Save the embeddings per topic label.
        if show_progress:
            print("Creating the topic's embeddings...")
        topic_labels = cluster_labels.labels_
        topic_label_embeds = {}
        for label, doc_embed in zip(topic_labels, doc_embeddings):
            topic_id = int(label)
            # Skip Noise labels:
            if topic_id == -1:
                continue
            # Check if this the first time we find this topic.
            if topic_id not in topic_label_embeds:
                topic_label_embeds[topic_id] = [doc_embed]
            else:
                topic_label_embeds[topic_id].append(doc_embed)

        # Progress bar variables.
        count = 0
        total = len(topic_label_embeds)
        # Find the average embed per label.
        topic_embeddings = {}
        for topic_label, label_embeds in topic_label_embeds.items():
            # Use Numpy to get the average embedding.
            mean_embeds = np.mean(label_embeds, axis=0)
            topic_embeddings[topic_label] = mean_embeds

            if show_progress:
                count += 1
                progress_bar(count, total)

        # The embeddings of the topics.
        return topic_embeddings

    def _find_doc_topics(self, show_progress=False):
        """
        Create a dictionary assigning each document to their closest topic.

        Args:
            show_progress: A Bool representing whether we show the progress of
                the function or not.

        Returns: A dictionary with the topic ID as key, and the list of documents
            belonging to the topics as value.
        """
        # Check we have at least a topic.
        if self.num_topics == 0:
            return {}

        # Progress Variables.
        count = 0
        total = len(self.doc_embeds)

        # Iterate through the documents and their embeddings.
        topic_documents = {}
        for doc_id, doc_embed in self.doc_embeds.items():
            # Find the closest topic to the document.
            topic_id, similarity = self._closest_topic(doc_embed)
            # Check if we have found this topic before.
            if topic_id in topic_documents:
                topic_documents[topic_id].append((doc_id, similarity))
            else:
                topic_documents[topic_id] = [(doc_id, similarity)]
            # Show Progress
            if show_progress:
                count += 1
                progress_bar(count, total)

        # Sort the Topic's Documents by similarity.
        for tuple_doc_sim in topic_documents.values():
            tuple_doc_sim.sort(key=lambda doc_sim: doc_sim[1], reverse=True)
        return topic_documents

    def _find_word_topics(self, show_progress=False):
        """
        Assign each word in the vocabulary to its closest topic.

        Args:
            show_progress: A Bool representing whether we show the progress of
                the function or not.

        Returns: A dictionary containing the topic IDs as keys, and th elist of
            words belonging to the topic as values.
        """
        # Check we have at least a topic.
        if self.num_topics == 0:
            return {}

        # Progress Variables
        count = 0
        total = len(self.word_embeds)

        # Iterate through the words and their embeddings.
        topic_words = {}
        for word, word_embed in self.word_embeds.items():
            # Find the closest topic to the word.
            topic_id, similarity = self._closest_topic(word_embed)
            # Check if the topic is already in the dictionary.
            if topic_id in topic_words:
                topic_words[topic_id].append((word, similarity))
            else:
                topic_words[topic_id] = [(word, similarity)]
            # Show Progress.
            if show_progress:
                count += 1
                progress_bar(count, total)

        # Sort the words' lists using their similarity to their topic.
        for tuple_word_sim in topic_words.values():
            tuple_word_sim.sort(key=lambda word_tuple: word_tuple[1], reverse=True)
        return topic_words

    def _closest_topic(self, embedding):
        """
        Given the embedding of a document or a word, find the closest topic to
        this embedding using cosine similarity.

        Args:
            embedding: Numpy.ndarray with the vector of the word or document we
                want to classify.

        Returns:
            A tuple with the ID of the closest topic and its similarity to the
                'embedding'.
        """
        # Use closest_vector(embedding, vectors_dict).
        closest_topic, max_similarity = closest_vector(embedding, self.topic_embeds)
        return closest_topic, max_similarity

    def save(self, model_id: str = None, show_progress=False):
        """
        Save the topic model. The saved Topic Model can be loaded later using
        the 'model_id'.

        Args:
            model_id: String with the ID we are using to identify the topic
                model.
            show_progress: A Bool representing whether we show the progress of
                the function or not.
        """
        # Progress Variables.
        count = 0
        total = len(self.word_embeds) + len(self.doc_embeds) + len(self.topic_embeds)

        # Check the project data folders exist.
        if not isdir(self.data_folder):
            mkdir(self.data_folder)
        topic_models_path = join(self.data_folder, self.topic_models_folder)
        if not isdir(topic_models_path):
            mkdir(topic_models_path)

        # Create folder for this topic model using ID or default.
        if model_id:
            model_folder_id = self.topic_model_prefix + model_id
        else:
            model_folder_id = self.topic_model_prefix + 'default'
        model_folder_path = join(topic_models_path, model_folder_id)
        if not isdir(model_folder_path):
            mkdir(model_folder_path)

        # Create & Save Index Dictionary for basic attributes.
        if show_progress:
            print("Saving Topic Model basic attributes...")
        topic_model_index = {
            'model_type': self.model_type,
            'use_title_abstract': self.use_title_abstract,
            'num_topics': self.num_topics,
            'corpus_ids': self.corpus_ids,
            'topic_docs': self.topic_docs,
            'topic_words': self.topic_words,
        }
        # Create index path.
        model_index_path = join(model_folder_path, self.model_index_file)
        with open(model_index_path, 'w') as f:
            json.dump(topic_model_index, f)

        # Progress Saving Dictionaries.
        if show_progress:
            print("Saving embedding dictionaries of Topic Model...")

        # Save Word Embeddings, first transforming dict to python types.
        word_embeds_index = {}
        for word_key, word_embed in self.word_embeds.items():
            word_embeds_index[word_key] = word_embed.tolist()
            if show_progress:
                count += 1
                progress_bar(count, total)
        # Create Word index path.
        word_index_path = join(model_folder_path, self.word_embeds_file)
        with open(word_index_path, 'w') as f:
            json.dump(word_embeds_index, f)

        # Save Doc Embeddings, transforming the dict to default python type.
        doc_embeds_index = {}
        for doc_key, doc_embed in self.doc_embeds.items():
            doc_embeds_index[doc_key] = doc_embed.tolist()
            if show_progress:
                count += 1
                progress_bar(count, total)
        # Create Doc Index path.
        doc_index_path = join(model_folder_path, self.doc_embeds_file)
        with open(doc_index_path, 'w') as f:
            json.dump(doc_embeds_index, f)

        # Save Topics Embeddings, transforming dict to default python type.
        topic_embeds_index = {}
        for topic_key, topic_embed in self.topic_embeds.items():
            topic_embeds_index[topic_key] = topic_embed.tolist()
            if show_progress:
                count += 1
                progress_bar(count, total)
        # Create Topic Index path.
        topic_index_path = join(model_folder_path, self.topic_embeds_file)
        with open(topic_index_path, 'w') as f:
            json.dump(topic_embeds_index, f)

    @classmethod
    def load(cls, model_id: str = None, show_progress=False):
        """
        Load a previously saved Topic Model.

        Args:
            model_id: String with the ID we are using to identify the topic
                model.
            show_progress: A Bool representing whether we show the progress of
                the function or not.
        Returns:
            An instance of the TopicModel class.
        """
        return cls(_used_saved=True, _model_id=model_id, show_progress=show_progress)


def closest_vector(embedding, vectors_dict):
    """
    Given the embedding of a document or word and a dictionary containing a
    group of vectors with their embeddings. Find the closest vector to the given
    embedding using cosine similarity.

    Args:
        embedding: Numpy.ndarray with the embedding of the word or document that
            we are using to find the closest vector.
        vectors_dict: Dictionary containing the vectors with the vectors IDs as
            keys and their embeddings as values.

    Returns:
        A tuple with the ID of the closest vector and its similarity to the
            'embedding' we received as parameter.
    """
    # Use iter to get the vectors IDs and their embeddings.
    vector_iter = iter(vectors_dict.items())

    # Get cosine similarity to the first vector.
    closest_vector_id, vector_embed = next(vector_iter)
    max_similarity = float(util.cos_sim(embedding, vector_embed))

    # Iterate through the rest of the vectors.
    for vector_id, vector_embed in vector_iter:
        new_similarity = float(util.cos_sim(embedding, vector_embed))
        if new_similarity > max_similarity:
            # New Closer Vector
            closest_vector_id = vector_id
            max_similarity = new_similarity

    # The closest vector ID with its similarity to the 'embedding'.
    return closest_vector_id, max_similarity


def cosine_similarity(a: np.ndarray, b: np.ndarray):
    """
    Calculate the cosine similarity between the vectors 'a' and 'b'.

    Args:
        a: Numpy.ndarray containing one of the vectors embeddings.
        b: Numpy.ndarray containing one of the vectors embeddings.

    Returns:
        Float with the cosine similarity between the two vectors.
    """
    # Use Numpy.
    result = np.dot(a, b) / (norm(a) * norm(b))
    return result


if __name__ == '__main__':
    # Record the Runtime of the Program
    stopwatch = TimeKeeper()

    # --Test TopicModel class--

    # Load Random Sample to use a limited amount of papers in CORD-19.
    test_size = 1_000
    print(f"\nLoading Random Sample of {big_number(test_size)} documents...")
    # sample = RandomSample(paper_type='medium', sample_size=test_size, show_progress=True)
    sample = RandomSample.load(show_progress=True)
    # Load RandomSample() saved with an id.
    # sample = RandomSample.load(sample_id='10000_docs', show_progress=True)
    print("Done.")
    print(f"[{stopwatch.formatted_runtime()}]")

    # # Use the entire CORD-19 Dataset
    # print("\nLoading the CORD-19 Dataset...")
    # sample = Papers(show_progress=True)
    # print("Done.")
    # print(f"[{stopwatch.formatted_runtime()}]")

    # Confirm amount of papers in the corpus.
    papers_count = len(sample.papers_cord_uids())
    print(f"\n{big_number(papers_count)} papers loaded.")

    # Use BERT Document Model.
    print("\nLoading Bert Model...")
    # bert_name = 'all-MiniLM-L12-v2'
    bert_name = 'paraphrase-MiniLM-L3-v2'
    my_model = BertCord19(model_name=bert_name, show_progress=True)

    # Use Specter Document Model.
    # print("\nLoading Specter model...")
    # my_model = SpecterManager(show_progress=True)

    # Use Doc2Vec Model trained with Cord-19 papers.
    # print("\nLoading Doc2Vec model of the Cord-19 Dataset...")
    # my_model = Doc2VecCord19.load('cord19_dataset', show_progress=True)

    print("Done.")
    print(f"[{stopwatch.formatted_runtime()}]")

    print("\nLoading Topic Model...")
    topic_model = TopicModel(corpus=sample, doc_model=my_model, only_title_abstract=True, show_progress=True)
    # topic_model = TopicModel.load(show_progress=True)
    print("Done.")
    print(f"[{stopwatch.formatted_runtime()}]")

    number_topics = topic_model.num_topics
    print(f"\n{number_topics} topics found.")

    print("\nTopics and Document count:")
    all_topics = topic_model.top_topics()
    for topic in all_topics:
        print(topic)

    top_n = 15
    print(f"\nTop {top_n} words per topic:")
    words_per_topic = topic_model.all_topics_top_words(top_n)
    for i, word_list in words_per_topic:
        print(f"\n----> Topic <{i}>:")
        for word_sim in word_list:
            print(word_sim)

    # print("\nSaving Topic Model...")
    # topic_model.save(show_progress=True)
    # print("Done.")
    # print(f"[{stopwatch.formatted_runtime()}]")
    #
    # print("\nLoading saved Topic Model...")
    # saved_model = TopicModel.load(show_progress=True)
    # print("Done.")
    # print(f"[{stopwatch.formatted_runtime()}]")
    #
    # print(f"\n{saved_model.num_topics} topics in saved Topic Model.")
    # print("\nTopics and Document Count:")
    # all_topics = saved_model.top_topics()
    # for topic in all_topics:
    #     print(topic)

    print("\nDone.")
    print(f"[{stopwatch.formatted_runtime()}]\n")
