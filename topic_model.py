# Gelin Eguinosa Rosique
# 2022

import json
import umap
import hdbscan
import multiprocessing
import numpy as np
from os import mkdir, listdir
from shutil import rmtree
from os.path import isdir, isfile, join
from collections import defaultdict
from numpy.linalg import norm

from corpus_cord19 import CorpusCord19
from papers_cord19 import PapersCord19
from document_model import DocumentModel
from random_sample import RandomSample
from bert_cord19 import BertCord19
from doc2vec_cord19 import Doc2VecCord19
from doc_tokenizers import doc_tokenizer
from extra_funcs import progress_bar, progress_msg, big_number
from time_keeper import TimeKeeper


# The Core Multiplier to calculate the Chunk sizes when doing Parallelism.
PARALLEL_MULT = 2
MAX_CORES = 8
PEAK_SIZE = 150


class TopicModel:
    """
    Find the topics in the CORD-19 corpus using the method described in Top2Vec.
    """
    # Class Data Locations.
    data_folder = 'project_data'
    class_data_folder = 'topic_models'
    model_folder_prefix = 'topics_'
    default_model_id = 'default'
    basic_index_file = 'topic_model_basic_index.json'
    model_index_file = 'topic_model_index.json'
    word_embeds_file = 'topic_model_word_embeds.json'
    doc_embeds_file = 'topic_model_doc_embeds.json'
    topic_embeds_file = 'topic_model_topic_embeds.json'
    reduced_topics_folder = 'reduced_topic_models'
    reduced_topic_prefix = 'reduced_topic_model_'

    # The Supported Doc Models.
    supported_doc_models = ['doc2vec', 'glove', 'bert', 'specter']

    def __init__(self, corpus: CorpusCord19 = None, doc_model: DocumentModel = None,
                 only_title_abstract=False, model_id=None, used_saved=False,
                 parallelism=False, show_progress=False):
        """
        Find the topics in the provided 'corpus' using 'doc_model' to get the
        embedding of the Documents and Words in the CORD-19 corpus selected.
            - If '_used_saved' is True, loads a previously used and saved model
              depending on the value of '_saved_id'.
            - If no ID is provided in 'saved_id', load the last used model.
            - If no 'corpus' is provided, use PapersCord19().
            - If no 'doc_model' is provided, use BertCord19().

        Args:
            corpus: A Cord-19 Corpus class with a selection of papers.
            doc_model: A Document Model class used to get the embeddings of the
                words and documents in the corpus.
            only_title_abstract: A Bool showing if we are going to use only the
                Titles & Abstracts of the papers, or all their content to create
                the vocabulary and the embeddings.
            model_id: String with the ID that identifies the Topic Model.
            used_saved: A Bool to know if we need to load the Topic Model from
                a file or recalculate it.
            parallelism: Bool to indicate if we can use parallelism to create
                the topics and create the topics' documents and vocabulary.
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
            topic_models_path = join(self.data_folder, self.class_data_folder)
            if not isdir(topic_models_path):
                raise FileNotFoundError("There is no Topic Model saved.")

            # Create folder path for this topic model.
            model_folder_id = self.model_folder_prefix + self.model_id
            model_folder_path = join(topic_models_path, model_folder_id)

            # Load Index Dictionary for basic attributes.
            if show_progress:
                progress_msg("Loading Topic Model basic attributes...")
            model_index_path = join(model_folder_path, self.model_index_file)
            if not isfile(model_index_path):
                raise FileNotFoundError("There is no Attributes Index available.")
            with open(model_index_path, 'r') as f:
                topic_model_index = json.load(f)
            # Get the Attributes.
            self.model_type = topic_model_index['model_type']
            self.num_topics = topic_model_index['num_topics']
            self.corpus_ids = topic_model_index['corpus_ids']
            self.use_title_abstract = topic_model_index['use_title_abstract']
            # Chance Dictionary's keys from str back to int keys (JSON issue).
            loaded_topic_docs = topic_model_index['topic_docs']
            self.topic_docs = dict([(int(key), docs)
                                    for key, docs in loaded_topic_docs.items()])
            loaded_topic_words = topic_model_index['topic_words']
            self.topic_words = dict([(int(key), words)
                                     for key, words in loaded_topic_words.items()])

            # Load Word Embeddings Dictionary.
            if show_progress:
                progress_msg("Loading Topic Model's word embeddings...")
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
                progress_msg("Loading Topic Model's doc embeddings...")
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
                progress_msg("Loading Topic's Embeddings...")
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
                self.topic_embeds[int(topic_key)] = np.array(topic_embed)
                if show_progress:
                    count += 1
                    progress_bar(count, total)
        else:
            # -- Create Topic Model --
            # Make sure we have a CORD-19 Corpus.
            if not corpus:
                corpus = PapersCord19()
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

            # Create the embeddings of the documents and find the topics.
            if show_progress:
                progress_msg("Creating Document Embeddings...")
            self.doc_embeds = self._create_docs_embeddings(corpus, doc_model, show_progress=show_progress)
            if show_progress:
                progress_msg("Finding Topics...")
            self.topic_embeds = self._find_topics(show_progress=show_progress)
            self.num_topics = len(self.topic_embeds)
            if show_progress:
                progress_msg(f"{self.num_topics} topics found.")
            if show_progress:
                progress_msg("Organizing documents by topics...")
            self.topic_docs = find_child_embeddings(parent_embeds=self.topic_embeds,
                                                    child_embeds=self.doc_embeds,
                                                    parallelism=parallelism,  # conflict with huggingface/tokenizers
                                                    show_progress=show_progress)
            # Create the embeddings of the Words to make the topics' vocabulary.
            if show_progress:
                progress_msg("Creating Word Embeddings...")
            self.word_embeds = self._create_word_embeddings(corpus, doc_model, show_progress=show_progress)
            if show_progress:
                progress_msg("Creating topics vocabulary...")
            self.topic_words = find_child_embeddings(parent_embeds=self.topic_embeds,
                                                     child_embeds=self.word_embeds,
                                                     parallelism=parallelism,  # conflict with huggingface/tokenizers
                                                     show_progress=show_progress)

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

    def generate_new_topics(self, number_topics: int, parallelism=False, show_progress=False):
        """
        Create a new Hierarchical Topic Model with specified number of topics
        (num_topics). The 'num_topics' need to be at least 2 topics, and be
        smaller than the original number of topics found.

        Args:
            number_topics: The desired topic count for the new Topic Model.
            parallelism: Bool indicating if we have to use multiprocessing or not.
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
                progress_msg("Invalid number of topics requested. No"
                             " hierarchical topic reduction performed.")
            # Exit function.
            return

        # Check if we can use a previously calculated Reduced Topic Model.
        use_saved_model = self.reduced_topics_saved()
        if not use_saved_model:
            # Initialize New Topics Variables.
            current_num_topics = self.num_topics
            new_topic_embeds = self.topic_embeds.copy()
            new_topic_sizes = dict([(topic_id, len(self.topic_docs[topic_id]))
                                    for topic_id in self.topic_docs.keys()])
        else:
            # Get the closest viable Reduced Topic Size.
            main_sizes = best_midway_sizes(self.num_topics)
            usable_sizes = [size for size in main_sizes if size >= number_topics]
            optimal_size = min(usable_sizes)
            # Upload The Reduced Topic Model.
            if show_progress:
                progress_msg(f"Loading Reduced Topic Model with {optimal_size} topics...")
            model_folder_name = self.model_folder_prefix + self.model_id
            reduced_topic_file = self.reduced_topic_prefix + str(optimal_size) + '.json'
            reduced_topic_path = join(self.data_folder, self.class_data_folder,
                                      model_folder_name, self.reduced_topics_folder,
                                      reduced_topic_file)
            with open(reduced_topic_path, 'r') as f:
                reduced_topic_index = json.load(f)
            # Get the Reduced Topic's Sizes.
            loaded_topic_sizes = reduced_topic_index['topic_sizes']
            new_topic_sizes = dict([(int(key), size)
                                    for key, size in loaded_topic_sizes.items()])
            # Get the Reduced Topic's Embeddings.
            json_topic_embeds = reduced_topic_index['topic_embeds']
            new_topic_embeds = {}
            for topic_id, topic_embed in json_topic_embeds.items():
                new_topic_embeds[int(topic_id)] = np.array(topic_embed)
            # Update Current Topic Size.
            current_num_topics = len(new_topic_embeds)

        # Perform topic reduction until we get the desired number of topics.
        while number_topics < current_num_topics:
            # Reduce the number of topics by 1.
            if show_progress:
                progress_msg(f"Reducing from {current_num_topics} to {current_num_topics - 1} topics...")
            result_tuple = self._reduce_topic_size(ref_topic_embeds=new_topic_embeds,
                                                   topic_sizes=new_topic_sizes,
                                                   parallelism=parallelism,
                                                   show_progress=show_progress)
            new_topic_embeds, new_topic_sizes = result_tuple
            # Update Current Number of Topics.
            current_num_topics = len(new_topic_embeds)

        # Progress - Done with the reduction of Topics.
        if show_progress:
            progress_msg(f"No need to reduce the current {current_num_topics} topics.")

        # Update New Topics' Attributes.
        self.new_topics = True
        self.new_num_topics = current_num_topics
        # Reset IDs of the New Topics.
        self.new_topic_embeds = dict([(new_id, topic_embed)
                                      for new_id, topic_embed
                                      in enumerate(new_topic_embeds.values())])
        # Assign Words and Documents to the New Topics.
        if show_progress:
            progress_msg("Organizing documents using the New Topics...")
        self.new_topic_docs = find_child_embeddings(self.new_topic_embeds,
                                                    self.doc_embeds,
                                                    parallelism=parallelism,
                                                    show_progress=show_progress)
        if show_progress:
            progress_msg("Creating the vocabulary for the New Topics...")
        self.new_topic_words = find_child_embeddings(self.new_topic_embeds,
                                                     self.word_embeds,
                                                     parallelism=parallelism,
                                                     show_progress=show_progress)
        # Assign Original Topics to the New Topics.
        if show_progress:
            progress_msg("Assigning original topics to the New topics...")
        self.topics_hierarchy = find_child_embeddings(self.new_topic_embeds,
                                                      self.topic_embeds,
                                                      parallelism=parallelism,
                                                      show_progress=show_progress)

    def _reduce_topic_size(self, ref_topic_embeds: dict, topic_sizes: dict,
                           parallelism=False, show_progress=False):
        """
        Reduce the provided Topics in 'ref_topic_embeds' by 1, mixing the
        smallest topic with its closest neighbor.

        Args:
            ref_topic_embeds: Dictionary containing the embeddings of the topics
                we are going to reduce. This dictionary is treated as a
                reference and will be modified to store the new reduced topics.
            topic_sizes: Dictionary containing the current size of the topics we
                are reducing.
            parallelism: Bool to indicate if we have to use the multiprocessing
                version of this function.
            show_progress: A Bool representing whether we show the progress of
                the function or not.

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
        if show_progress:
            progress_msg(f"Creating sizes for the new {len(ref_topic_embeds)} topics...")
        new_topic_sizes = self._topic_document_count(ref_topic_embeds,
                                                     parallelism=parallelism,
                                                     show_progress=show_progress)
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

    def _create_word_embeddings(self, corpus: CorpusCord19, doc_model: DocumentModel,
                                show_progress=False):
        """
        Create a dictionary with all the words in the corpus and their
        embeddings.

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
            new_words = [word for word in doc_tokens if word not in words_embeddings]
            new_embeds = doc_model.words_vectors(new_words)
            for new_word, new_embed in zip(new_words, new_embeds):
                words_embeddings[new_word] = new_embed
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
        # Default dict to store the doc's embeddings.
        doc_embeddings = {}
        # Batch Size to process documents in groups. (Speeding Up)
        batch_size = max(1, len(self.corpus_ids) // 100)

        # Progress Bar variables.
        count = 0
        total = len(self.corpus_ids)
        # Process the documents in batches to speed up process.
        processed_docs = 0
        total_docs = len(self.corpus_ids)
        batch_count = 0
        batch_ids = []
        batch_docs = []
        for cord_uid in self.corpus_ids:
            # Get Doc's content depending on the model type.
            if self.model_type == 'specter':
                # Use the doc ID to get their pre-calculated embedding.
                doc_content = cord_uid
            elif self.use_title_abstract or self.model_type == 'bert':
                # Bert only support a limited amount of word count.
                doc_content = corpus.paper_title_abstract(cord_uid)
            elif self.model_type in {'doc2vec', 'glove'}:
                # The previous case includes when Doc2Vec and title_abstract.
                doc_content = corpus.paper_content(cord_uid)
            else:
                # Using a not supported model.
                raise NameError(f"We don't support the Model<{self.model_type}> yet.")

            # Add new ID and document to the batch.
            batch_count += 1
            processed_docs += 1
            batch_ids.append(cord_uid)
            batch_docs.append(doc_content)

            # See if this the last document or the batch is full.
            if batch_count == batch_size or processed_docs == total_docs:
                # Get the encodings of the documents.
                new_embeddings = doc_model.documents_vectors(batch_docs)
                # Add new embeddings to dictionary.
                for new_id, new_embed in zip(batch_ids, new_embeddings):
                    # Skip documents with empty encodings.
                    if not np.any(new_embed):
                        continue
                    doc_embeddings[new_id] = new_embed
                # Reset batch list and counter.
                batch_count = 0
                batch_ids = []
                batch_docs = []

            # Progress.
            if show_progress:
                count += 1
                progress_bar(count, total)

        # The dictionary with the Docs IDs and their embeddings.
        return doc_embeddings

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
            progress_msg("UMAP: Reducing dimensions of the papers...")
        umap_embeddings = umap.UMAP(n_neighbors=15,
                                    n_components=5,
                                    metric='cosine').fit_transform(doc_embeddings)

        # Use HDBSCAN to find the cluster of documents in the vector space.
        if show_progress:
            progress_msg("HDBSCAN: Creating topic clusters with the documents...")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=15,
                                    metric='euclidean',
                                    cluster_selection_method='eom')
        cluster_labels = clusterer.fit(umap_embeddings)

        # Save the embeddings per topic label.
        if show_progress:
            progress_msg("Creating the topic's embeddings...")
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

    def _topic_document_count(self, topic_embeds_dict: dict, parallelism=False,
                              show_progress=False):
        """
        Given a dictionary with the embeddings of a group of topics, count the
        number of documents assign to each of the topics in the given dictionary.

        Args:
            topic_embeds_dict: Dictionary with the topic IDs as keys and the
                embeddings of the topics as values.
            parallelism: Bool to indicate if we have to use the multiprocessing
                version of this function.
            show_progress: A Bool representing whether we show the progress of
                the function or not.

        Returns:
            Dictionary containing the topic IDs as keys and the number of
                documents belonging to each one in the current corpus.
        """
        # Check if multiprocessing was requested, and we have enough topics.
        # parallel_min = int(2 * PEAK_SIZE / MAX_CORES)  # I made a formula that game me this number (?)
        parallel_min = 37  # This is the number when more cores are faster.
        if parallelism and len(topic_embeds_dict) > parallel_min:
            return self._document_count_parallel(topic_embeds_dict, show_progress)

        # Check we have at least a topic.
        if len(topic_embeds_dict) == 0:
            return {}

        # Progress Variables.
        count = 0
        total = len(self.doc_embeds)
        # Iterate through the documents and their embeddings.
        topic_docs_count = {}
        for doc_id, doc_embed in self.doc_embeds.items():
            # Find the closest topic to the current document.
            topic_id, _ = closest_vector(doc_embed, topic_embeds_dict)
            # Check if we have found this topic before.
            if topic_id in topic_docs_count:
                topic_docs_count[topic_id] += 1
            else:
                topic_docs_count[topic_id] = 1
            # Show Progress:
            if show_progress:
                count += 1
                progress_bar(count, total)

        # The document count per each topic.
        return topic_docs_count

    def _document_count_parallel(self, topic_embeds_dict: dict, show_progress=False):
        """
        Version of _topic_document_count() using MultiProcessing.

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

        # Determine the number of cores.
        optimal_cores = min(multiprocessing.cpu_count(), MAX_CORES)
        efficiency_mult = min(float(1), len(topic_embeds_dict) / PEAK_SIZE)
        core_count = max(2, int(efficiency_mult * optimal_cores))
        # Number of instructions processed in one batch.
        chunk_size = max(1, len(self.doc_embeds) // 100)
        # chunk_size = max(1, len(self.doc_embeds) // (PARALLEL_MULT * core_count))

        # Dictionary for the Topic-Docs count.
        topic_docs_count = defaultdict(int)
        # Create parameter tuples for _custom_closest_vector().
        tuple_params = [(doc_id, doc_embed, topic_embeds_dict)
                        for doc_id, doc_embed in self.doc_embeds.items()]
        with multiprocessing.Pool(processes=core_count) as pool:
            # Pool map() vs imap() depending on if we have to report progress.
            if show_progress:
                # Report Parallelization.
                progress_msg(f"Using Parallelization <{core_count} cores>")
                # Progress Variables.
                count = 0
                total = len(self.doc_embeds)
                # Iterate through the results to update the process.
                for doc_id, topic_id, _ in pool.imap(_custom_closest_vector, tuple_params, chunksize=chunk_size):
                    topic_docs_count[topic_id] += 1
                    # Progress.
                    count += 1
                    progress_bar(count, total)
            else:
                # Process all the parameters at once (faster).
                tuple_results = pool.map(_custom_closest_vector, tuple_params, chunksize=chunk_size)
                for doc_id, topic_id, _ in tuple_results:
                    topic_docs_count[topic_id] += 1

        # The document count per each topic.
        topic_docs_count = dict(topic_docs_count)
        return topic_docs_count

    def save(self, model_id: str = None, show_progress=False):
        """
        Save the topic model. The saved Topic Model can be loaded later using
        the 'model_id'.

        If a new 'model_id' is provided the ID of the current Topic Model is
        updated.

        Args:
            model_id: String with the ID we want to use to identify the file of
                the topic model. Overrides the value of the Model's current ID.
            show_progress: A Bool representing whether we show the progress of
                the function or not.
        """
        # Check if we need to update the ID of the Model.
        if model_id:
            self.model_id = model_id

        # Progress Variables.
        count = 0
        total = len(self.word_embeds) + len(self.doc_embeds) + len(self.topic_embeds)

        # Check the project data folders exist.
        if not isdir(self.data_folder):
            mkdir(self.data_folder)
        class_folder_path = join(self.data_folder, self.class_data_folder)
        if not isdir(class_folder_path):
            mkdir(class_folder_path)
        # Create Folder Model Folder Path.
        model_folder_name = self.model_folder_prefix + self.model_id
        model_folder_path = join(class_folder_path, model_folder_name)
        # Check if there is already a model saved with this name.
        if isdir(model_folder_path):
            # Delete the previously saved model with this name.
            rmtree(model_folder_path)
        # Create New Empty Folder for the model.
        mkdir(model_folder_path)

        # Save Topic Model's Basic Info.
        self.save_basic_info()

        # Create & Save Index Dictionary for basic attributes.
        if show_progress:
            progress_msg("Saving Topic Model basic attributes...")
        topic_model_index = {
            'model_type': self.model_type,
            'num_topics': self.num_topics,
            'corpus_ids': self.corpus_ids,
            'topic_docs': self.topic_docs,
            'topic_words': self.topic_words,
            'use_title_abstract': self.use_title_abstract,
        }
        # Create index path.
        model_index_path = join(model_folder_path, self.model_index_file)
        with open(model_index_path, 'w') as f:
            json.dump(topic_model_index, f)

        # Progress Saving Dictionaries.
        if show_progress:
            progress_msg("Saving embedding dictionaries of Topic Model...")

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

    def save_basic_info(self):
        """
        Save the number of topics, the number of documents and the Document
        Model Type of current Topic Model inside the Model's Folder.
        """
        # Check the project data folders exist.
        if not isdir(self.data_folder):
            mkdir(self.data_folder)
        class_folder_path = join(self.data_folder, self.class_data_folder)
        if not isdir(class_folder_path):
            mkdir(class_folder_path)
        model_folder_name = self.model_folder_prefix + self.model_id
        model_folder_path = join(class_folder_path, model_folder_name)
        if not isdir(model_folder_path):
            mkdir(model_folder_path)

        # Save index with the basic data.
        basic_index = {
            'model_type': self.model_type,
            'num_topics': self.num_topics,
            'corpus_size': len(self.corpus_ids),
            'topics_hierarchy': self.reduced_topics_saved(),
        }
        # Create path.
        basic_index_path = join(model_folder_path, self.basic_index_file)
        with open(basic_index_path, 'w') as f:
            json.dump(basic_index, f)

    def save_reduced_topics(self, parallelism=False, show_progress=False):
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
            parallelism: Bool to indicate if we have to use the multiprocessing
                version of this function.
            show_progress:  A Bool representing whether we show the progress of
                the function or not.
        """
        # Check we can create a Reduced Topic Model.
        if self.num_topics <= 2:
            return

        # Check the class' data folders.
        if not isdir(self.data_folder):
            mkdir(self.data_folder)
        class_folder_path = join(self.data_folder, self.class_data_folder)
        if not isdir(class_folder_path):
            mkdir(class_folder_path)
        model_folder_name = self.model_folder_prefix + self.model_id
        model_folder_path = join(class_folder_path, model_folder_name)
        if not isdir(model_folder_path):
            mkdir(model_folder_path)
        # Check if there is an already Saved Hierarchy.
        reduced_folder_path = join(model_folder_path, self.reduced_topics_folder)
        if isdir(reduced_folder_path):
            # Remove the previously saved Hierarchy.
            rmtree(reduced_folder_path)
        # Create Empty Folder to store the hierarchically reduced topics.
        mkdir(reduced_folder_path)

        # Get a Set with the Reduced Topic Sizes that we have to save.
        main_sizes = best_midway_sizes(self.num_topics)

        # Initialize Topic Reduction dictionaries.
        current_num_topics = self.num_topics
        new_topic_embeds = self.topic_embeds.copy()
        new_topic_sizes = dict([(topic_id, len(self.topic_docs[topic_id]))
                                for topic_id in self.topic_docs.keys()])

        # Start Reducing Topics and Saving the Main Topic Sizes.
        if show_progress:
            progress_msg("Saving Main Hierarchically Reduced Topic Models...")
        while current_num_topics > 2:
            # Reduce number of topics by 1.
            if show_progress:
                progress_msg(f"Reducing from {current_num_topics} to {current_num_topics - 1} topics...")
            result_tuple = self._reduce_topic_size(ref_topic_embeds=new_topic_embeds,
                                                   topic_sizes=new_topic_sizes,
                                                   parallelism=parallelism,
                                                   show_progress=show_progress)
            new_topic_embeds, new_topic_sizes = result_tuple
            # Update current number of topics.
            current_num_topics = len(new_topic_embeds)

            # Check if we need to save the current embeddings and sizes.
            if current_num_topics in main_sizes:
                if show_progress:
                    progress_msg("<<Main Topic Found>>")
                    progress_msg(f"Saving Reduced Topic Model with {current_num_topics} topics...")
                # Transform Embeddings to lists.
                json_topic_embeds = {}
                for topic_id, topic_embed in new_topic_embeds.items():
                    json_topic_embeds[topic_id] = topic_embed.tolist()
                # Create Dict with embeddings and sizes.
                reduced_topic_index = {
                    'topic_embeds': json_topic_embeds,
                    'topic_sizes': new_topic_sizes,
                }
                # Save the index of the reduced topic.
                reduced_topic_file = (self.reduced_topic_prefix
                                      + str(current_num_topics) + '.json')
                reduced_topic_path = join(reduced_folder_path, reduced_topic_file)
                with open(reduced_topic_path, 'w') as f:
                    json.dump(reduced_topic_index, f)
                # Progress.
                if show_progress:
                    progress_msg("<<Saved>>")

        # Update the Basic Info of the Model (It has now a Hierarchy).
        self.save_basic_info()

    def reduced_topics_saved(self):
        """
        Check if the main Hierarchically Reduced Topics were properly saved.

        Returns: Bool showing if the Reduced Topic Models were saved.
        """
        # Check the Model's Folders.
        if not isdir(self.data_folder):
            return False
        class_folder_path = join(self.data_folder, self.class_data_folder)
        if not isdir(class_folder_path):
            return False
        model_folder_name = self.model_folder_prefix + self.model_id
        model_folder_path = join(class_folder_path, model_folder_name)
        if not isdir(model_folder_path):
            return False
        reduced_folder_path = join(model_folder_path, self.reduced_topics_folder)
        if not isdir(reduced_folder_path):
            return False

        # Check that all the Main Reduced Topic Models were saved.
        main_sizes = best_midway_sizes(self.num_topics)
        for topic_size in main_sizes:
            # Check the file for the Reduced Model with the current size.
            reduced_topic_file = self.reduced_topic_prefix + str(topic_size) + '.json'
            reduced_topic_path = join(reduced_folder_path, reduced_topic_file)
            if not isfile(reduced_topic_path):
                return False

        # All The Files were created correctly.
        return True

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

    @classmethod
    def basic_info(cls, model_id: str = None):
        """
        Load the Basic Model Info of the Topic Model 'model_id'.

        Args:
            model_id: String with the ID of the Topic Model.

        Returns:
            Dictionary with the 'model_type', 'num_topics', 'corpus_size' and
                the 'topics_hierarchy' bool of the Topic Model.
        """
        # If no 'model_id' is provided load the Default Model.
        if not model_id:
            model_id = cls.default_model_id

        # Check the existence of the data folders.
        if not isdir(cls.data_folder):
            raise FileNotFoundError("There is no Data Folder available.")
        class_folder_path = join(cls.data_folder, cls.class_data_folder)
        if not isdir(class_folder_path):
            raise FileNotFoundError("There is no Class Data Folder available.")
        model_folder_name = cls.model_folder_prefix + model_id
        model_folder_path = join(class_folder_path, model_folder_name)
        if not isdir(model_folder_path):
            raise FileNotFoundError(f"The Topic Model <{model_id}> has no Data Folder.")

        # Load the Basic Index.
        basic_index_path = join(model_folder_path, cls.basic_index_file)
        if not isfile(basic_index_path):
            raise FileNotFoundError(f"The Topic Model <{model_id}> has not Basic Index File.")
        with open(basic_index_path, 'r') as f:
            basic_index = json.load(f)

        # The Basic Index of the Topic Model.
        return basic_index

    @classmethod
    def saved_topic_models(cls):
        """
        Create a list with the IDs of the saved Topic Models.

        Returns: List[String] with the IDs of the saved Topic Models.
        """
        # Check the Data Folders.
        if not isdir(cls.data_folder):
            return []
        class_folder_path = join(cls.data_folder, cls.class_data_folder)
        if not isdir(class_folder_path):
            return []

        # Check all the available IDs inside class folder.
        topic_ids = []
        for entry_name in listdir(class_folder_path):
            entry_path = join(class_folder_path, entry_name)
            # Check we have a valid Model Folder Name.
            if not isdir(entry_path):
                continue
            if not entry_name.startswith(cls.model_folder_prefix):
                continue

            # Check for the Index Files.
            basic_index_path = join(entry_path, cls.basic_index_file)
            if not isfile(basic_index_path):
                continue
            model_index_path = join(entry_path, cls.model_index_file)
            if not isfile(model_index_path):
                continue
            word_embeds_path = join(entry_path, cls.word_embeds_file)
            if not isfile(word_embeds_path):
                continue
            doc_embeds_path = join(entry_path, cls.doc_embeds_file)
            if not isfile(doc_embeds_path):
                continue
            topic_embeds_path = join(entry_path, cls.topic_embeds_file)
            if not isfile(topic_embeds_path):
                continue

            # Save Model ID, the folder contains all the main indexes.
            model_id = entry_name.replace(cls.model_folder_prefix, '', 1)
            topic_ids.append(model_id)

        # List with the IDs of all the valid Topic Models.
        return topic_ids


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
                          parallelism=False, show_progress=False):
    """
    Given a 'parent_embeds' embeddings dictionary and a 'child_embeds'
    embeddings dictionary, create a new dictionary assigning each of the
    child_ids to their closest parent_id in the embedding space.

    Args:
        parent_embeds: Dictionary containing the parent_ids as keys and their
            embeddings as values.
        child_embeds: Dictionary containing the child_ids as keys and their
            embeddings as values.
        parallelism: Use the multiprocessing version of this function.
        show_progress: A Bool representing whether we show the progress of
            the function or not.

    Returns:
        Dictionary containing the parent_ids as keys, and a List of the closest
        child_ids to them in the embedding space as values.
    """
    # See if we have to use the multiprocessing version.
    # parallel_min = int(2 * PEAK_SIZE / MAX_CORES)  # I made a formula that game me this number (?)
    parallel_min = 37  # This is the number when more cores are faster.
    if parallelism and len(parent_embeds) > parallel_min:
        return find_children_parallel(parent_embeds, child_embeds, show_progress)

    # Check if we have at least one Parent Dictionary.
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


def find_children_parallel(parent_embeds: dict, child_embeds: dict,
                           show_progress=False):
    """
    A version of find_child_embeddings() using parallelism.

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
    # Check if we have at least one Parent Dictionary.
    if len(parent_embeds) == 0:
        return {}

    # Determine the number of cores to be used.
    optimal_cores = min(multiprocessing.cpu_count(), MAX_CORES)
    efficiency_mult = min(float(1), len(parent_embeds) / PEAK_SIZE)
    core_count = max(2, int(efficiency_mult * optimal_cores))
    # Create chunk size to process the tasks in the cores.
    chunk_size = max(1, len(child_embeds) // 100)
    # chunk_size = max(1, len(child_embeds) // (PARALLEL_MULT * core_count))
    # Create tuple parameters.
    tuple_params = [(child_id, child_embed, parent_embeds)
                    for child_id, child_embed in child_embeds.items()]

    # Create default Parent-Children dictionary.
    parent_child_dict = defaultdict(list)
    # Create the Processes Pool using all the available CPUs.
    with multiprocessing.Pool(processes=core_count) as pool:
        # Using Pool.imap(), we have to show the progress.
        if show_progress:
            # Report Parallelization.
            progress_msg(f"Using Parallelization <{core_count} cores>")
            # Progress Variables.
            count = 0
            total = len(child_embeds)
            # Find the closest parent to each child.
            for child_id, parent_id, similarity in pool.imap(_custom_closest_vector,
                                                             tuple_params,
                                                             chunksize=chunk_size):
                # Add the child to its closest parent.
                parent_child_dict[parent_id].append((child_id, similarity))
                # Progress.
                if show_progress:
                    count += 1
                    progress_bar(count, total)
        # No need to report progress, use Pool.map()
        else:
            # Find the closest parent to each child.
            tuple_results = pool.map(_custom_closest_vector, tuple_params, chunksize=chunk_size)
            # Save the closest children to each Parent.
            for child_id, parent_id, similarity in tuple_results:
                parent_child_dict[parent_id].append((child_id, similarity))

    # Sort Children's List by their similarity to their parents.
    parent_child_dict = dict(parent_child_dict)
    for tuples_child_sim in parent_child_dict.values():
        tuples_child_sim.sort(key=lambda child_sim: child_sim[1], reverse=True)
    return parent_child_dict


def _custom_closest_vector(id_dicts_tuple):
    """
    Custom-made version of the method closest_vector to use in the method
    find_children_parallel(), also a version of the find_child_embeddings()
    function.

    Take the 'child_id' and 'child_embed', with the parent_embeds inside the
    'id_dicts_tuple' parameter, and find the closest parent embedding to the
    child embedding with the provided 'child_id'.

    Args:
        id_dicts_tuple: Tuple containing
            ('child_id', 'child_embed', 'parent_embeds') to call the function
            closest_vector().

    Returns:
        Tuple with the 'child_id', its closest 'parent_id' and their 'similarity'.
    """
    child_id, child_embed, parent_embeds = id_dicts_tuple
    parent_id, similarity = closest_vector(child_embed, parent_embeds)
    return child_id, parent_id, similarity


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
    midway_sizes.update(range(5, min(30, original_size), 5))
    # Sizes between 30 and 100.
    midway_sizes.update(range(30, min(100, original_size), 10))
    # Sizes between 100 and 300.
    midway_sizes.update(range(100, min(300, original_size), 25))
    # Sizes between 300 and 1000.
    midway_sizes.update(range(300, min(1_001, original_size), 50))

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
    topic_corpus = PapersCord19(show_progress=True)
    print("Done.")
    print(f"[{time_record.formatted_runtime()}]")
    # Report Corpus size.
    corpus_size = len(topic_corpus.papers_cord_uids())
    print(f"\n{big_number(corpus_size)} papers loaded.")

    # Create Topic Model using Doc2Vec (Small Model).
    print("\nLoading Doc2Vec Small Model...")
    doc2vec_type = 'cord19_title_abstract'
    doc_model = Doc2VecCord19.load(model_id=doc2vec_type, show_progress=True)
    print("Done.")
    print(f"[{time_record.formatted_runtime()}]")
    # Create Topic Model.
    topic_model_id = 'cord19_dataset_doc2vec_small'
    print("\nCreating Topic Model using the Small Doc2Vec Model...")
    topic_model = TopicModel(corpus=topic_corpus, doc_model=doc_model,
                             only_title_abstract=True, show_progress=True)
    print("Done.")
    print(f"[{time_record.formatted_runtime()}]")
    # Report Topics found.
    print(f"\n{topic_model.num_topics} topics found.")
    # Save Topic Model.
    print(f"\nSaving Topic Model with ID: {topic_model_id}")
    topic_model.save(topic_model_id, show_progress=True)
    print("Done.")
    print(f"[{time_record.formatted_runtime()}]")

    # Create Topic Model using Doc2Vec (Big Model).
    print("\nLoading Doc2Vec Big Model...")
    doc2vec_type = 'cord19_dataset'
    doc_model = Doc2VecCord19.load(model_id=doc2vec_type, show_progress=True)
    print("Done.")
    print(f"[{time_record.formatted_runtime()}]")
    # Create Topic Model.
    topic_model_id = 'cord19_dataset_doc2vec_big'
    print("\nCreating Topic Model using the Big Doc2Vec Model...")
    topic_model = TopicModel(corpus=topic_corpus, doc_model=doc_model,
                             only_title_abstract=True, show_progress=True)
    print("Done.")
    print(f"[{time_record.formatted_runtime()}]")
    # Report Topics found.
    print(f"\n{topic_model.num_topics} topics found.")
    # Save Topic Model.
    print(f"\nSaving Topic Model with ID: {topic_model_id}")
    topic_model.save(topic_model_id, show_progress=True)
    print("Done.")
    print(f"[{time_record.formatted_runtime()}]")

    # Create Topic Model using GloVe.
    print("\nLoading BERT-GloVe Model...")
    bert_type = 'average_word_embeddings_glove.6B.300d'
    doc_model = BertCord19(model_name=bert_type, show_progress=True)
    print("Done.")
    print(f"[{time_record.formatted_runtime()}]")
    # Create Topic Model.
    topic_model_id = 'cord19_dataset_glove'
    print("\nCreating Topic Model using GloVe...")
    topic_model = TopicModel(corpus=topic_corpus, doc_model=doc_model,
                             only_title_abstract=True, show_progress=True)
    print("Done.")
    print(f"[{time_record.formatted_runtime()}]")
    # Report Topics found.
    print(f"\n{topic_model.num_topics} topics found.")
    # Save Topic Model.
    print(f"\nSaving Topic Model with ID: {topic_model_id}")
    topic_model.save(topic_model_id, show_progress=True)
    print("Done.")
    print(f"[{time_record.formatted_runtime()}]")

    # Create Topic Model using Bert (fast version).
    print("\nLoading Fastest BERT Model...")
    bert_type = 'paraphrase-MiniLM-L3-v2'
    doc_model = BertCord19(model_name=bert_type, show_progress=True)
    print("Done.")
    print(f"[{time_record.formatted_runtime()}]")
    # Create Topic Model.
    topic_model_id = 'cord19_dataset_bert_fast'
    print("\nCreating Topic Model using BERT (fastest version)...")
    topic_model = TopicModel(corpus=topic_corpus, doc_model=doc_model,
                             only_title_abstract=True, show_progress=True)
    print("Done.")
    print(f"[{time_record.formatted_runtime()}]")
    # Report Topics found.
    print(f"\n{topic_model.num_topics} topics found.")
    # Save Topic Model.
    print(f"\nSaving Topic Model with ID: {topic_model_id}")
    topic_model.save(topic_model_id, show_progress=True)
    print("Done.")
    print(f"[{time_record.formatted_runtime()}]")

    # Create Topic Model using Bert Multilingual (fast version).
    print("\nLoading Fast Multilingual BERT Model...")
    bert_type = 'paraphrase-multilingual-MiniLM-L12-v2'
    doc_model = BertCord19(model_name=bert_type, show_progress=True)
    print("Done.")
    print(f"[{time_record.formatted_runtime()}]")
    # Create Topic Model.
    topic_model_id = 'cord19_dataset_bert_multilingual'
    print("\nCreating Topic Model using Multilingual BERT (fast version)...")
    topic_model = TopicModel(corpus=topic_corpus, doc_model=doc_model,
                             only_title_abstract=True, show_progress=True)
    print("Done.")
    print(f"[{time_record.formatted_runtime()}]")
    # Report Topics found.
    print(f"\n{topic_model.num_topics} topics found.")
    # Save Topic Model.
    print(f"\nSaving Topic Model with ID: {topic_model_id}")
    topic_model.save(topic_model_id, show_progress=True)
    print("Done.")
    print(f"[{time_record.formatted_runtime()}]")

    # Create Topic Model using Bert (best performing).
    print("\nLoading Best Performing BERT Model...")
    bert_type = 'all-mpnet-base-v2'
    doc_model = BertCord19(model_name=bert_type, show_progress=True)
    print("Done.")
    print(f"[{time_record.formatted_runtime()}]")
    # Create Topic Model.
    topic_model_id = 'cord19_dataset_bert_best'
    print("\nCreating Topic Model using the Best Performing BERT...")
    topic_model = TopicModel(corpus=topic_corpus, doc_model=doc_model,
                             only_title_abstract=True, show_progress=True)
    print("Done.")
    print(f"[{time_record.formatted_runtime()}]")
    # Report Topics found.
    print(f"\n{topic_model.num_topics} topics found.")
    # Save Topic Model.
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
    # topic_model_id = 'cord19_dataset_specter'
    # print("\nCreating Topic Model using SPECTER...")
    # topic_model = TopicModel(corpus=topic_corpus, doc_model=doc_model,
    #                          only_title_abstract=True, show_progress=True)
    # print("Done.")
    # print(f"[{time_record.formatted_runtime()}]")
    # # Report Topics found.
    # print(f"\n{topic_model.num_topics} topics found.")
    # # Save Topic Model.
    # print(f"\nSaving Topic Model with ID: {topic_model_id}")
    # topic_model.save(topic_model_id, show_progress=True)
    # print("Done.")
    # print(f"[{time_record.formatted_runtime()}]")

    print("\nDone.")
    print(f"[{time_record.formatted_runtime()}]")


if __name__ == '__main__':
    # Record the Runtime of the Program
    stopwatch = TimeKeeper()

    # -- Load Corpus --
    # # Load Random Sample to use a limited amount of papers from CORD-19.
    # test_size = 500
    # print(f"\nLoading Random Sample of {big_number(test_size)} documents...")
    # sample = RandomSample(paper_type='medium', sample_size=test_size, show_progress=True)
    # sample = RandomSample.load(show_progress=True)

    # Load RandomSample() saved with an id.
    sample_id = '5000_docs'
    print(f"\nLoading Saved Random Sample <{sample_id}>...")
    sample = RandomSample.load(sample_id=sample_id, show_progress=True)
    # ---------------------------------------------
    # # Use CORD-19 Dataset
    # print("\nLoading the CORD-19 Dataset...")
    # sample = PapersCord19(show_progress=True)
    print("Done.")
    print(f"[{stopwatch.formatted_runtime()}]")

    # -- Report amount of papers in the loaded Corpus --
    papers_count = len(sample.papers_cord_uids())
    print(f"\n{big_number(papers_count)} papers loaded.")

    # -- Load Document Model --
    # Use BERT Document Model.
    print("\nLoading Bert Model...")
    # bert_name = 'all-MiniLM-L12-v2'
    bert_name = 'paraphrase-MiniLM-L3-v2'  # Fastest model.
    my_model = BertCord19(model_name=bert_name, show_progress=True)
    # ---------------------------------------------
    # # Use Specter Document Model.
    # print("\nLoading Specter model...")
    # my_model = SpecterManager(show_progress=True)
    # ---------------------------------------------
    # Use Doc2Vec Model trained with Cord-19 papers.
    # print("\nLoading Doc2Vec model of the Cord-19 Dataset...")
    # my_model = Doc2VecCord19.load('cord19_dataset', show_progress=True)
    print("Done.")
    print(f"[{stopwatch.formatted_runtime()}]")

    # -- Load Topic Model --
    # the_model_id = 'testing_' + my_model.model_type()
    the_model_id = f'test_{my_model.model_type()}_{sample_id}'
    # ---------------------------------------------
    # Creating Topic Model.
    print(f"\nCreating Topic Model with ID <{the_model_id}>...")
    the_topic_model = TopicModel(corpus=sample, doc_model=my_model, only_title_abstract=True,
                                 model_id=the_model_id, show_progress=True)
    # ---------------------------------------------
    # print(f"Saving Topic Model with ID <{the_topic_model.model_id}>...")
    # # new_model_id = the_model_id
    # new_model_id = the_model_id + f"_{the_topic_model.num_topics}topics"
    # the_topic_model.save(model_id=new_model_id, show_progress=True)
    # ---------------------------------------------
    # # Loading Saved Topic Model.
    # the_model_id = 'test_bert_25000_docs_196topics_parallel'
    # print(f"\nLoading Topic Model <{the_model_id}>...")
    # the_topic_model = TopicModel.load(model_id=the_model_id, show_progress=True)
    # ---------------------------------------------
    progress_msg("Done.")
    print(f"[{stopwatch.formatted_runtime()}]")

    # -- Show the Topics Created --
    total_topics = the_topic_model.num_topics
    print(f"\n{total_topics} topics found.")
    # ---------------------------------------------
    print("\nTopics and Document count:")
    all_topics = the_topic_model.top_topics()
    for topic_and_size in all_topics:
        print(topic_and_size)
    # ---------------------------------------------
    # # Topics' Vocabulary
    # # top_n = 15
    # # print(f"\nTop {top_n} words per topic:")
    # # words_per_topic = the_topic_model.all_topics_top_words(top_n)
    # # for i, word_list in words_per_topic:
    # #     print(f"\n----> Topic <{i}>:")
    # #     for word_sim in word_list:
    # #         print(word_sim)

    # # --Test Creating Hierarchically Reduced Topics--
    # # Save the Hierarchically Reduced Topic Models.
    # print("\nSaving Topic Model's Topic Hierarchy...")
    # the_topic_model.save_reduced_topics(parallelism=True, show_progress=True)

    # # -- Show Hierarchically Reduced Topics --
    # new_topics = 10
    # print(f"\nCreating Topic Model with {new_topics} topics.")
    # the_topic_model.generate_new_topics(number_topics=new_topics, show_progress=True)
    # print("Done.")
    # print(f"[{stopwatch.formatted_runtime()}]")
    #
    # print("\nNew Topics and Document count:")
    # all_topics = the_topic_model.top_topics()
    # for topic in all_topics:
    #     print(topic)
    #
    # top_n = 15
    # print(f"\nTop {top_n} words per new topic:")
    # words_per_topic = the_topic_model.all_topics_top_words(top_n)
    # for i, word_list in words_per_topic:
    #     print(f"\n----> Topic <{i}>:")
    #     for word_sim in word_list:
    #         print(word_sim)

    # # --**-- Create Hierarchy for CORD-19 Topic Models --**--
    # the_model_id = 'cord19_dataset_bert_fast'
    # print(f"\nLoading Topic Model with ID <{the_model_id}>...")
    # the_topic_model = TopicModel.load(model_id=the_model_id, show_progress=True)
    # print("Done.")
    # print(f"[{stopwatch.formatted_runtime()}]")
    #
    # total_topics = the_topic_model.num_topics
    # print(f"\n{total_topics} topics found.")
    #
    # print("\nTopics and Document count:")
    # all_topics = the_topic_model.top_topics()
    # for tuple_topic_size in all_topics:
    #     print(tuple_topic_size)
    #
    # # Save the Hierarchically Reduced Topic Models.
    # print("\nSaving Topic Model's Topic Hierarchy...")
    # the_topic_model.save_reduced_topics(show_progress=True)

    print("\nDone.")
    print(f"[{stopwatch.formatted_runtime()}]\n")
