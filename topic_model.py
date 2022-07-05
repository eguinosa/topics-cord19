# Gelin Eguinosa Rosique

import json
import umap
import hdbscan
import numpy as np
from os import mkdir
from os.path import isdir, isfile, join
from numpy.linalg import norm

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
    topic_model_prefix = 'topics_'
    default_model_id = 'default'
    model_index_file = 'topic_model_index.json'
    word_embeds_file = 'topic_model_word_embeds.json'
    doc_embeds_file = 'topic_model_doc_embeds.json'
    topic_embeds_file = 'topic_model_topic_embeds.json'

    # The Doc Models available to use.
    supported_doc_models = ['doc2vec', 'glove', 'bert', 'specter']

    def __init__(self, corpus: CorpusCord19 = None, doc_model: DocumentModel = None,
                 only_title_abstract=False, model_id=None, used_saved=False,
                 show_progress=False):
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
            model_id: A string with the ID of a previously saved Topic Model.
            used_saved: A Bool to know if we need to load the Topic Model from
                a file or recalculate it.
            show_progress: Bool representing whether we show the progress of
                the function or not.
        """
        # Save ID of the file of the Topic Model.
        self.model_id = model_id if model_id else self.default_model_id

        if used_saved:
            # -- Load Topic Model ---
            # Check if the data folders exist.
            if not isdir(self.data_folder):
                raise FileNotFoundError("There is no project data available.")
            topic_models_path = join(self.data_folder, self.topic_models_folder)
            if not isdir(topic_models_path):
                raise FileNotFoundError("There is no Topic Model saved.")

            # Create folder path for this topic model.
            model_folder_id = self.topic_model_prefix + self.model_id
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

            # *************For Now:
            # Needs Update!!!!
            # Create Default values for topics created with a fixed number of
            # desired topics:
            # ----------------------------------------------------------------
            # Bool indicating if we have hierarchically reduced topics.
            self.new_topics = False
            # Create attributes for the Hierarchical Topic Reduction.
            self.new_num_topics = None
            self.new_topic_embeds = None
            self.new_topic_docs = None
            self.new_topic_words = None
            # Dictionary with new topics as keys, and their closest original
            # topics as values.
            self.topics_hierarchy = None
        else:
            # -- Create Topic Model --
            # Make sure we have a CORD-19 Corpus.
            if not corpus:
                corpus = Papers()
            self.corpus_ids = corpus.papers_cord_uids()

            # Check we have a Document Model.
            if not doc_model:
                doc_model = BertCord19()
            self.model_type = doc_model.model_type()

            # Save if we are only using the Title and Abstract of the Papers.
            self.use_title_abstract = only_title_abstract

            # Check we are using a supported model.
            if self.model_type not in self.supported_doc_models:
                raise NameError(f"The model type <{self.model_type}> is not supported.")

            # Calculate the embeddings of the words and documents.
            if show_progress:
                print("Creating Word Embeddings...")
            self.word_embeds = self._create_word_embeddings(corpus, doc_model, show_progress=show_progress)
            if show_progress:
                print("Creating Document Embeddings...")
            self.doc_embeds = self._create_docs_embeddings(corpus, doc_model, show_progress=show_progress)
            if show_progress:
                print("Finding Topics...")
            self.topic_embeds = self._find_topics(show_progress=show_progress)
            self.num_topics = len(self.topic_embeds)
            if show_progress:
                print(f"{self.num_topics} topics found.")
            if show_progress:
                print("Organizing documents by topics...")
            self.topic_docs = find_child_embeddings(self.topic_embeds, self.doc_embeds)
            if show_progress:
                print("Creating topics vocabulary...")
            self.topic_words = find_child_embeddings(self.topic_embeds, self.word_embeds)

            # Create Default values for topics created with a fixed number of
            # desired topics:
            # ----------------------------------------------------------------
            # Bool indicating if we have hierarchically reduced topics.
            self.new_topics = False
            # Create attributes for the Hierarchical Topic Reduction.
            self.new_num_topics = None
            self.new_topic_embeds = None
            self.new_topic_docs = None
            self.new_topic_words = None
            # Dictionary with new topics as keys, and their closest original
            # topics as values.
            self.topics_hierarchy = None

    def generate_new_topics(self, number_topics: int, show_progress=False):
        """
        Create a new Hierarchical Topic Model with specified number of topics
        (num_topics). The 'num_topics' need to be at least 2 topics, and be
        smaller than the original number of topics found.

        Args:
            number_topics: The desired topic count for the new Topic Model.
            show_progress: Bool representing whether we show the progress of
                the function or not.
        """
        # Check the number of topics requested is valid.
        if not 1 < number_topics < self.num_topics:
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
        total = current_num_topics - number_topics
        # Perform topic reduction until we get the desired number of topics
        if show_progress:
            print(f"Reducing from {self.num_topics} to {number_topics} topics.")
        while number_topics < current_num_topics:
            # Reduce the number of topics by 1.
            new_topic_embeds, new_topic_sizes = self._reduce_topic_size(new_topic_embeds, new_topic_sizes)
            # Update Current Number of Topics.
            current_num_topics = len(new_topic_embeds)
            # Show progress.
            if show_progress:
                count += 1
                progress_bar(count, total)

        # Update New Topics' Attributes.
        self.new_topics = True
        self.new_num_topics = current_num_topics
        # Reset IDs of the New Topics.
        self.new_topic_embeds = dict([(new_id, topic_embed)
                                      for new_id, topic_embed
                                      in enumerate(new_topic_embeds.values())])
        # Assign Words and Documents to the New Topics.
        if show_progress:
            print("Organizing documents using the New Topics...")
        self.new_topic_docs = find_child_embeddings(self.new_topic_embeds,
                                                    self.doc_embeds,
                                                    show_progress=show_progress)
        if show_progress:
            print("Creating the vocabulary for the New Topics...")
        self.new_topic_words = find_child_embeddings(self.new_topic_embeds,
                                                     self.word_embeds,
                                                     show_progress=show_progress)
        # Assign Original Topics to the New Topics.
        if show_progress:
            print("Assigning original topics to the New topics...")
        self.topics_hierarchy = find_child_embeddings(self.new_topic_embeds,
                                                      self.topic_embeds,
                                                      show_progress=show_progress)

    def _reduce_topic_size(self, ref_topic_embeds: dict, topic_sizes: dict):
        """
        Reduce the provided Topics in 'ref_topic_embeds' by 1, mixing the
        smallest topic with its closest neighbor.

        Args:
            ref_topic_embeds: Dictionary containing the embeddings of the topics
                we are going to reduce. This dictionary is treated as a
                reference and will be modified to store the new reduced topics.
            topic_sizes: Dictionary containing the current size of the topics we
                are reducing.

        Returns:
            Tuple with 'ref_topic_embeds' dictionary  and a new 'topic_sizes'
                dictionary containing the updated embeddings and sizes
                respectively for the new Topics.
        """
        # Get the smallest topic and its info.
        new_topics_list = list(ref_topic_embeds.keys())
        min_topic_id = min(new_topics_list, key=lambda x: len(self.topic_docs[x]))
        min_embed = ref_topic_embeds[min_topic_id]

        # Delete Smallest Topic.
        del ref_topic_embeds[min_topic_id]
        # Get the closest topic to the small topic.
        close_topic_id, _ = closest_vector(min_embed, ref_topic_embeds)
        close_embed = ref_topic_embeds[close_topic_id]

        # Merge the embedding of the topics.
        min_size = topic_sizes[min_topic_id]
        close_size = topic_sizes[close_topic_id]
        total_size = min_size + close_size
        merged_topic_embed = (min_size * min_embed + close_size * close_embed) / total_size

        # Update embedding of the closest topic.
        ref_topic_embeds[close_topic_id] = merged_topic_embed
        # Get the new topic sizes.
        new_topic_sizes = self._topic_document_count(ref_topic_embeds)
        # New Dictionaries with embeds and sizes.
        return ref_topic_embeds, new_topic_sizes

    def top_topics(self, show_originals=False):
        """
        Make a sorted list with the topics organized by the amount of documents
        they represent.

        Args:
            show_originals: Bool to indicate if we need to use the original
                topics even if we already have found New Topics.

        Returns:
            A list of tuples with the topics' ID and their document count.
        """
        # Choose between the original topics or the hierarchical reduced.
        if show_originals or not self.new_topics:
            topic_docs_source = self.topic_docs
        else:
            topic_docs_source = self.new_topic_docs

        # Form list of topics with their size.
        topic_docs = [(topic_id, len(docs_list))
                      for topic_id, docs_list in topic_docs_source.items()]
        # Sort by size.
        topic_docs.sort(key=lambda count: count[1], reverse=True)
        return topic_docs

    def top_words_topic(self, topic_id, num_words=10, show_originals=False):
        """
        Find the top n words for the given topic. If 'num_words' is -1, then
        return all the words belonging to this topic.

        Args:
            topic_id: The topic from which we want the top words.
            num_words: The amount of words from that topic that we want.
            show_originals: Bool to indicate if we need to use the original
                topics even if we already have found New Topics.

        Returns:
            A list of tuples with the words and their similarity to the topic.
        """
        # Choose between the original topics or the hierarchical reduced.
        if show_originals or not self.new_topics:
            topic_words_source = self.topic_words
        else:
            topic_words_source = self.new_topic_words

        # Check the topic exists.
        if topic_id not in topic_words_source:
            raise NameError("Topic not found.")

        # Get the list of words we are returning.
        if num_words == -1:
            # All words in the topic.
            result = topic_words_source[topic_id]
        else:
            # Check we are not giving more words than what we can.
            word_count = min(num_words, len(topic_words_source[topic_id]))
            # Get the number of words requested.
            result = topic_words_source[topic_id][:word_count]
        # List of tuples with words and similarities.
        return result

    def all_topics_top_words(self, num_words=10, show_originals=False):
        """
        Make a list with the top words per topics. If 'num_words' is -1, returns
        all the words belonging to a topic.

        Args:
            num_words: The amount of words we want from each topic.
            show_originals: Bool to indicate if we need to use the original
                topics even if we already have found New Topics.

        Returns:
            A list of tuples with the Topic ID and their top_words_topic(),
                the latter containing the top words for the corresponding topic.
        """
        # Check if we are using the original topics or the hierarchical reduced.
        if show_originals or not self.new_topics:
            show_orishas = True
        else:
            show_orishas = False

        # Create list of words per topic.
        topics_top_words = []
        for topic_id, _ in self.top_topics(show_originals=show_orishas):
            topic_top_words = self.top_words_topic(topic_id,
                                                   num_words=num_words,
                                                   show_originals=show_orishas)
            topics_top_words.append((topic_id, topic_top_words))

        # Per topic, a list with the words and their similarity to the topic.
        return topics_top_words

    def _create_word_embeddings(self, corpus: CorpusCord19,
                                doc_model: DocumentModel, show_progress=False):
        """
        Create a dictionary with all the words in the corpus and their embeddings.

        Args:
            corpus: A Cord-19 Corpus class with the selection of papers in the
                corpus.
            doc_model: A Document Model class used to get the embeddings of the
                words in the corpus.
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
            content_provider = corpus.all_papers_title_abstract()
        else:
            content_provider = corpus.all_papers_content()

        # Create the vocabulary using the tokens from the Corpus' Content.
        words_embeddings = {}
        for doc_content in content_provider:
            doc_tokens = doc_tokenizer(doc_content)
            # Add the new words from the document.
            for doc_word in doc_tokens:
                if doc_word not in words_embeddings:
                    word_embed = doc_model.word_vector(doc_word)
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

    def _create_docs_embeddings(self, corpus: CorpusCord19,
                                doc_model: DocumentModel, show_progress=False):
        """
        Calculate the embeddings of the documents using the specified Document
        Model.

        Args:
            corpus: A Cord-19 Corpus class with the selection of papers in the
                corpus.
            doc_model: A Document Model class used to get the embeddings of the
                documents in the corpus.
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
                doc_embedding = doc_model.document_vector(cord_uid)
            elif self.use_title_abstract or self.model_type == 'bert':
                # BERT has a length limit, use only title and abstract.
                doc_content = corpus.paper_title_abstract(cord_uid)
                doc_embedding = doc_model.document_vector(doc_content)
            elif self.model_type in {'doc2vec', 'glove'}:
                # No text size or token count restrictions, use all text.
                doc_content = corpus.paper_content(cord_uid)
                doc_embedding = doc_model.document_vector(doc_content)
            else:
                raise NameError(f"We don't support the Model<{self.model_type}> yet.")

            # Save Doc Embedding, skipping Docs that the model can't encode.
            if np.any(doc_embedding):
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

    def _topic_document_count(self, topic_embeds_dict: dict, show_progress=False):
        """
        Given a dictionary with the embeddings of a group of topics, count the
        number of documents assign to each of the topics in the given dictionary.

        Args:
            topic_embeds_dict: Dictionary with the topic IDs as keys and the
                embeddings of the topics as values.
            show_progress: A Bool representing whether we show the progress of
                the function or not.

        Returns:
            Dictionary containing the topic IDs as keys and the number of
                documents belonging to each one in the current corpus.
        """
        # Check we have at least a topic.
        if len(topic_embeds_dict) == 0:
            return {}

        # Progress Variables.
        count = 0
        total = len(topic_embeds_dict)

        # Iterate through the documents and their embeddings.
        topic_docs_count = {}
        for doc_id, doc_embed in self.doc_embeds.items():
            # Find the closest topic to the current document.
            topic_id, _ = closest_vector(doc_embed, topic_embeds_dict)
            # Check if we have found this topic before.
            if topic_id in topic_docs_count:
                topic_docs_count[topic_id] += 1
            else:
                topic_docs_count[topic_id] = 0
            # Show Progress:
            if show_progress:
                count += 1
                progress_bar(count, total)

        # The document count per each topic.
        return topic_docs_count

    def save(self, model_id: str = None, show_progress=False):
        """
        Save the topic model. The saved Topic Model can be loaded later using
        the 'model_id'.

        Args:
            model_id: String with the ID we want to use to identify the file of
                the topic model. Overrides the value of the Model's current ID.
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
            model_folder_id = self.topic_model_prefix + self.model_id
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

    def _save_reduced_topics(self, model_folder_path: str, show_progress=False):
        """
        Create a list of basic topic sizes between 2 and the size of the current
        Topic Model, to create and save the Hierarchical Topic Models of this
        Model with these sizes, so when we create a new Hierarchical Topic Model
        we can do it faster, only having to start reducing the Topic sizes from
        the closest basic topic size.

        The saved topic sizes will be in the range of 2-1000, with different
        steps depending on the Topic Size range.
        - Step of  5 between  2 and 30.
        - Step of 10 between 30 and 100.
        - Step of 25 between 100 and 300.
        - Step of 50 between 300 and 1000.

        Args:
            model_folder_path: String with the path to the folder were the
                current Model's attributes are stored.
            show_progress:  A Bool representing whether we show the progress of
                the function or not.
        """
        # Check if we have a valid Folder Path:
        if not isdir(model_folder_path):
            raise NotADirectoryError("The provided Model Folder is not valid.")
        # Check we can create a Reduced Topic Model.
        if self.num_topics <= 2:
            return

        # Get a Set with the Reduced Topic Sizes that we have to save.
        main_sizes = best_midway_sizes(self.num_topics)

        # Initialize Topic Reduction Variables.
        new_topic_embeds = self.topic_embeds.copy()
        new_topic_sizes = dict([(topic_id, len(self.topic_docs[topic_id]))
                                for topic_id in self.topic_docs.keys()])
        # Not Done yet...

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
        return cls(model_id=model_id, used_saved=True, show_progress=show_progress)


def closest_vector(embedding, vectors_dict: dict):
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
    max_similarity = cosine_similarity(embedding, vector_embed)

    # Iterate through the rest of the vectors.
    for vector_id, vector_embed in vector_iter:
        new_similarity = cosine_similarity(embedding, vector_embed)
        if new_similarity > max_similarity:
            # New Closer Vector
            closest_vector_id = vector_id
            max_similarity = new_similarity

    # The closest vector ID with its similarity to the 'embedding'.
    return closest_vector_id, max_similarity


def find_child_embeddings(parent_embeds: dict, child_embeds: dict,
                          show_progress=False):
    """
    Given a 'parent_embeds' embeddings dictionary and a 'child_embeds'
    embeddings dictionary, create a new dictionary assigning each of the
    child_ids to their closest parent_id in the embedding space.

    Args:
        parent_embeds: Dictionary containing the parent_ids as keys and their
            embeddings as values.
        child_embeds: Dictionary containing the child_ids as keys and their
            embeddings as values.
        show_progress: A Bool representing whether we show the progress of
            the function or not.

    Returns:
        Dictionary containing the parent_ids as keys, and a List of the closest
        child_ids to them in the embedding space as values.
    """
    # Check if we have at least a Parent Dictionary.
    if len(parent_embeds) == 0:
        return {}

    # Progress Variables.
    count = 0
    total = len(child_embeds)

    # Iterate through each of the children and assign them to their closest parent.
    parent_child_dict = {}
    for child_id, child_embed in child_embeds.items():
        # Find the closest parent to the child.
        parent_id, similarity = closest_vector(child_embed, parent_embeds)
        # Check if we have found this parent before.
        if parent_id in parent_child_dict:
            parent_child_dict[parent_id].append((child_id, similarity))
        else:
            parent_child_dict[parent_id] = [(child_id, similarity)]
        # Show Progress.
        if show_progress:
            count += 1
            progress_bar(count, total)

    # Sort Children's List by their similarity to their parents.
    for tuples_child_sim in parent_child_dict.values():
        tuples_child_sim.sort(key=lambda child_sim: child_sim[1], reverse=True)
    return parent_child_dict


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
    # Transform from float32 to float (float32 is not JSON serializable)
    result = float(result)
    return result


def best_midway_sizes(original_size: int):
    """
    Create a set containing the topic sizes for whom we are going to save a
    Hierarchically Reduced Topic Model to speed up the process of returning
    a Topic Model when a user requests a custom size.

    The Topic Sizes will have a different step depending on the range of topic
    sizes:
    - Step of  5 between  2 and 30.
    - Step of 10 between 30 and 100.
    - Step of 25 between 100 and 300.
    - Step of 50 between 300 and 1000.

    Example: 2, 5, 10, ..., 25, 30, 40,..., 90, 100, 125, ..., 275, 300, 350...

    Args:
        original_size: Int with the size of the original Topic Model.

    Returns:
        Set containing the Ints with the topic sizes we have to save when
            generating the reduced Topic Model.
    """
    # Check we don't have an Original Topic of size 2.
    midway_sizes = set()
    if original_size > 2:
        midway_sizes.add(2)
    # Sizes between 5 and 30.
    midway_sizes.add(range(5, min(30, original_size), 5))
    # Sizes between 30 and 100.
    midway_sizes.add(range(30, min(100, original_size), 10))
    # Sizes between 100 and 300.
    midway_sizes.add(range(100, min(300, original_size), 25))
    # Sizes between 300 and 1000.
    midway_sizes.add(range(300, min(1_001, original_size), 50))

    # The Intermediate sizes to create a reference Hierarchical Topic Model.
    return midway_sizes


def save_cord19_topics():
    """
    Create and Save the Topic Models for the CORD-19 dataset using Bert, Doc2Vec
    and Specter.
    """
    # Record the Runtime of the Program
    time_record = TimeKeeper()

    # Load Corpus.
    print("\nLoading the CORD-19 corpus...")
    topic_corpus = Papers(show_progress=True)
    print("Done.")
    print(f"[{time_record.formatted_runtime()}]")
    # Report Corpus size.
    corpus_size = len(topic_corpus.papers_cord_uids())
    print(f"\n{big_number(corpus_size)} papers loaded.")

    # # Create Topic Model using Doc2Vec (Small Model).
    # print("\nLoading Doc2Vec Small Model...")
    # doc2vec_type = 'cord19_title_abstract'
    # doc_model = Doc2VecCord19.load(model_id=doc2vec_type, show_progress=True)
    # print("Done.")
    # print(f"[{time_record.formatted_runtime()}]")
    # # Create Topic Model.
    # print("\nCreating Topic Model using the Small Doc2Vec Model...")
    # topic_model = TopicModel(corpus=topic_corpus, doc_model=doc_model,
    #                          only_title_abstract=True, show_progress=True)
    # print("Done.")
    # print(f"[{time_record.formatted_runtime()}]")
    # # Report Topics found.
    # print(f"\n{topic_model.num_topics} topics found.")
    # # Save Topic Model.
    # topic_model_id = 'cord19_dataset_doc2vec_small'
    # print(f"\nSaving Topic Model with ID: {topic_model_id}")
    # topic_model.save(topic_model_id, show_progress=True)
    # print("Done.")
    # print(f"[{time_record.formatted_runtime()}]")

    # # Create Topic Model using Doc2Vec (Big Model).
    # print("\nLoading Doc2Vec Big Model...")
    # doc2vec_type = 'cord19_dataset'
    # doc_model = Doc2VecCord19.load(model_id=doc2vec_type, show_progress=True)
    # print("Done.")
    # print(f"[{time_record.formatted_runtime()}]")
    # # Create Topic Model.
    # print("\nCreating Topic Model using the Big Doc2Vec Model...")
    # topic_model = TopicModel(corpus=topic_corpus, doc_model=doc_model,
    #                          only_title_abstract=True, show_progress=True)
    # print("Done.")
    # print(f"[{time_record.formatted_runtime()}]")
    # # Report Topics found.
    # print(f"\n{topic_model.num_topics} topics found.")
    # # Save Topic Model.
    # topic_model_id = 'cord19_dataset_doc2vec_big'
    # print(f"\nSaving Topic Model with ID: {topic_model_id}")
    # topic_model.save(topic_model_id, show_progress=True)
    # print("Done.")
    # print(f"[{time_record.formatted_runtime()}]")

    # # Create Topic Model using GloVe.
    # print("\nLoading BERT-GloVe Model...")
    # bert_type = 'average_word_embeddings_glove.6B.300d'
    # doc_model = BertCord19(model_name=bert_type, show_progress=True)
    # print("Done.")
    # print(f"[{time_record.formatted_runtime()}]")
    # # Create Topic Model.
    # print("\nCreating Topic Model using GloVe...")
    # topic_model = TopicModel(corpus=topic_corpus, doc_model=doc_model,
    #                          only_title_abstract=True, show_progress=True)
    # print("Done.")
    # print(f"[{time_record.formatted_runtime()}]")
    # # Report Topics found.
    # print(f"\n{topic_model.num_topics} topics found.")
    # # Save Topic Model.
    # topic_model_id = 'cord19_dataset_glove'
    # print(f"\nSaving Topic Model with ID: {topic_model_id}")
    # topic_model.save(topic_model_id, show_progress=True)
    # print("Done.")
    # print(f"[{time_record.formatted_runtime()}]")

    # # Create Topic Model using Bert (fast version).
    # print("\nLoading Fastest BERT Model...")
    # bert_type = 'paraphrase-MiniLM-L3-v2'
    # doc_model = BertCord19(model_name=bert_type, show_progress=True)
    # print("Done.")
    # print(f"[{time_record.formatted_runtime()}]")
    # # Create Topic Model.
    # print("\nCreating Topic Model using BERT (fastest version)...")
    # topic_model = TopicModel(corpus=topic_corpus, doc_model=doc_model,
    #                          only_title_abstract=True, show_progress=True)
    # print("Done.")
    # print(f"[{time_record.formatted_runtime()}]")
    # # Report Topics found.
    # print(f"\n{topic_model.num_topics} topics found.")
    # # Save Topic Model.
    # topic_model_id = 'cord19_dataset_bert_fast'
    # print(f"\nSaving Topic Model with ID: {topic_model_id}")
    # topic_model.save(topic_model_id, show_progress=True)
    # print("Done.")
    # print(f"[{time_record.formatted_runtime()}]")

    # # Create Topic Model using Bert Multilingual (fast version).
    # print("\nLoading Fast Multilingual BERT Model...")
    # bert_type = 'paraphrase-multilingual-MiniLM-L12-v2'
    # doc_model = BertCord19(model_name=bert_type, show_progress=True)
    # print("Done.")
    # print(f"[{time_record.formatted_runtime()}]")
    # # Create Topic Model.
    # print("\nCreating Topic Model using Multilingual BERT (fast version)...")
    # topic_model = TopicModel(corpus=topic_corpus, doc_model=doc_model,
    #                          only_title_abstract=True, show_progress=True)
    # print("Done.")
    # print(f"[{time_record.formatted_runtime()}]")
    # # Report Topics found.
    # print(f"\n{topic_model.num_topics} topics found.")
    # # Save Topic Model.
    # topic_model_id = 'cord19_dataset_bert_multilingual'
    # print(f"\nSaving Topic Model with ID: {topic_model_id}")
    # topic_model.save(topic_model_id, show_progress=True)
    # print("Done.")
    # print(f"[{time_record.formatted_runtime()}]")

    # Create Topic Model using Bert (best performing).
    print("\nLoading Best Performing BERT Model...")
    bert_type = 'all-mpnet-base-v2'
    doc_model = BertCord19(model_name=bert_type, show_progress=True)
    print("Done.")
    print(f"[{time_record.formatted_runtime()}]")
    # Create Topic Model.
    print("\nCreating Topic Model using the Best Performing BERT...")
    topic_model = TopicModel(corpus=topic_corpus, doc_model=doc_model,
                             only_title_abstract=True, show_progress=True)
    print("Done.")
    print(f"[{time_record.formatted_runtime()}]")
    # Report Topics found.
    print(f"\n{topic_model.num_topics} topics found.")
    # Save Topic Model.
    topic_model_id = 'cord19_dataset_bert_best'
    print(f"\nSaving Topic Model with ID: {topic_model_id}")
    topic_model.save(topic_model_id, show_progress=True)
    print("Done.")
    print(f"[{time_record.formatted_runtime()}]")

    # # Create Topic Model using Specter.
    # print("\nLoading SPECTER Model...")
    # doc_model = SpecterManager(load_full_dicts=True, show_progress=True)
    # print("Done.")
    # print(f"[{time_record.formatted_runtime()}]")
    # # Create Topic Model.
    # print("\nCreating Topic Model using SPECTER...")
    # topic_model = TopicModel(corpus=topic_corpus, doc_model=doc_model,
    #                          only_title_abstract=True, show_progress=True)
    # print("Done.")
    # print(f"[{time_record.formatted_runtime()}]")
    # # Report Topics found.
    # print(f"\n{topic_model.num_topics} topics found.")
    # # Save Topic Model.
    # topic_model_id = 'cord19_dataset_specter'
    # print(f"\nSaving Topic Model with ID: {topic_model_id}")
    # topic_model.save(topic_model_id, show_progress=True)
    # print("Done.")
    # print(f"[{time_record.formatted_runtime()}]")

    print("\nDone.")
    print(f"[{time_record.formatted_runtime()}]")


if __name__ == '__main__':
    # Test Saving all The Topic Models
    save_cord19_topics()

    # # Record the Runtime of the Program
    # stopwatch = TimeKeeper()
    #
    # # --Test TopicModel class--
    #
    # # Load Random Sample to use a limited amount of papers in CORD-19.
    # # test_size = 500
    # # print(f"\nLoading Random Sample of {big_number(test_size)} documents...")
    # # sample = RandomSample(paper_type='medium', sample_size=test_size, show_progress=True)
    # sample = RandomSample.load(show_progress=True)
    # # # Load RandomSample() saved with an id.
    # # sample = RandomSample.load(sample_id='10000_docs', show_progress=True)
    # print("Done.")
    # print(f"[{stopwatch.formatted_runtime()}]")
    #
    # # # Use the entire CORD-19 Dataset
    # # print("\nLoading the CORD-19 Dataset...")
    # # sample = Papers(show_progress=True)
    # # print("Done.")
    # # print(f"[{stopwatch.formatted_runtime()}]")
    #
    # # Confirm amount of papers in the corpus.
    # papers_count = len(sample.papers_cord_uids())
    # print(f"\n{big_number(papers_count)} papers loaded.")
    #
    # # Use BERT Document Model.
    # print("\nLoading Bert Model...")
    # # bert_name = 'all-MiniLM-L12-v2'
    # bert_name = 'paraphrase-MiniLM-L3-v2'
    # my_model = BertCord19(model_name=bert_name, show_progress=True)
    #
    # # # Use Specter Document Model.
    # # print("\nLoading Specter model...")
    # # my_model = SpecterManager(show_progress=True)
    #
    # # Use Doc2Vec Model trained with Cord-19 papers.
    # # print("\nLoading Doc2Vec model of the Cord-19 Dataset...")
    # # my_model = Doc2VecCord19.load('cord19_dataset', show_progress=True)
    #
    # print("Done.")
    # print(f"[{stopwatch.formatted_runtime()}]")
    #
    # print("\nLoading Topic Model...")
    # the_topic_model = TopicModel(corpus=sample, doc_model=my_model,
    #                              only_title_abstract=True, show_progress=True)
    # the_model_id = 'testing_bert_save'
    # print(f"Saving Topic Model with ID <{the_model_id}>")
    # the_topic_model.save(model_id=the_model_id, show_progress=True)
    # # topic_model = TopicModel.load(show_progress=True)
    # print("Done.")
    # print(f"[{stopwatch.formatted_runtime()}]")
    #
    # total_topics = the_topic_model.num_topics
    # print(f"\n{total_topics} topics found.")
    #
    # print("\nTopics and Document count:")
    # all_topics = the_topic_model.top_topics()
    # for topic in all_topics:
    #     print(topic)
    #
    # top_n = 15
    # print(f"\nTop {top_n} words per topic:")
    # words_per_topic = the_topic_model.all_topics_top_words(top_n)
    # for i, word_list in words_per_topic:
    #     print(f"\n----> Topic <{i}>:")
    #     for word_sim in word_list:
    #         print(word_sim)
    #
    # # # --Test Creating Hierarchically Reduced Topics--
    # # the_num_topics = 3
    # # print(f"\nCreating Topic Model with {the_num_topics} topics.")
    # # the_topic_model.generate_new_topics(number_topics=3, show_progress=True)
    # # print("Done.")
    # # print(f"[{stopwatch.formatted_runtime()}]")
    # #
    # # print("\nNew Topics and Document count:")
    # # all_topics = the_topic_model.top_topics()
    # # for topic in all_topics:
    # #     print(topic)
    # #
    # # top_n = 15
    # # print(f"\nTop {top_n} words per new topic:")
    # # words_per_topic = the_topic_model.all_topics_top_words(top_n)
    # # for i, word_list in words_per_topic:
    # #     print(f"\n----> Topic <{i}>:")
    # #     for word_sim in word_list:
    # #         print(word_sim)
    # #
    # # # --Test Saving Topic Model--
    # # print("\nSaving Topic Model...")
    # # the_topic_model.save(show_progress=True)
    # # print("Done.")
    # # print(f"[{stopwatch.formatted_runtime()}]")
    # #
    # # print("\nLoading saved Topic Model...")
    # # saved_model = TopicModel.load(show_progress=True)
    # # print("Done.")
    # # print(f"[{stopwatch.formatted_runtime()}]")
    # #
    # # print(f"\n{saved_model.num_topics} topics in saved Topic Model.")
    # # print("\nTopics and Document Count:")
    # # all_topics = saved_model.top_topics()
    # # for topic in all_topics:
    # #     print(topic)
    #
    # print("\nDone.")
    # print(f"[{stopwatch.formatted_runtime()}]\n")
