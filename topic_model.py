# Gelin Eguinosa Rosique

import numpy
import torch
import umap
import hdbscan
from sentence_transformers import util
from pprint import pprint

from random_sample import RandomSample
from bert_cord19 import BertCord19
from doc_tokenizers import doc_tokenizer
from extra_funcs import progress_bar
from time_keeper import TimeKeeper


class TopicModel:
    """
    Find the topics in the CORD-19 corpus using the method described in Top2Vec.
    """
    # Class Data Locations.
    data_folder = 'project_data'

    # The Doc Models available to use.
    doc_model_types = ['doc2vec', 'glove', 'bert', 'specter', 'longformer']

    def __init__(self, doc_type='medium', sample_size=-1, model_type='bert',
                 model_name='average_word_embeddings_glove.6B.300d',
                 show_progress=False):
        """
        Create a Random sample of CORD-19 papers with the specified 'doc_type'.
        Use the 'model_type' to create the embeddings of the words and documents,
        and with these embeddings find the topics present in the corpus.

        Args:
            doc_type: The type of Papers we want to use for the Sample (small,
                medium or big).
            sample_size: An int with size of the sample. The default value '-1'
                represents all the papers available with the specified paper
                type.
            model_type: The Document and Word Model we are going to use to
                create the embeddings.
            model_name: For some models like BERT or GloVe that have models
                of different sizes and speed, we need the name of the specific
                model we want to use.
            show_progress: A Bool representing whether we show the progress of
                the function or not.
        """
        # Check we are using a supported model.
        if model_type not in self.doc_model_types:
            raise NameError(f"The model type <{model_type}> is not supported.")
        # Save the model type.
        self.model_type = model_type

        # Load the CORD-19 sample we are going to use.
        if show_progress:
            print("Creating RandomSample()...")
        self.corpus_sample = RandomSample(paper_type=doc_type,
                                          sample_size=sample_size,
                                          show_progress=show_progress)
        # Save the Papers() class in case we want to access individual papers.
        self.corpus_manager = self.corpus_sample.cord19_papers
        # Also, the IDs of the Papers in the Sample.
        self.corpus_cord_uids = self.corpus_sample.sample_cord_uids

        # Load Document & Word Model (All in one class).
        # Using bert for now.
        if show_progress:
            print(f"Loading the Document Model <{model_type}>...")
        self.doc_model = BertCord19(model_name=model_name,
                                    show_progress=show_progress)

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
        self.topic_docs = self._find_doc_topics()
        self.topic_words = self._find_word_topics()
        self.num_topics = len(self.topic_embeds)

    def top_topics(self):
        """
        Make a sorted list with the topics organized by the amount of documents
        they represent.

        Returns:
            A list of tuples with the topics' ID and their document count.
        """
        topic_docs = [(top_id, len(docs_list)) for top_id, docs_list in self.topic_docs.items()]
        topic_docs.sort(key=lambda count: count[1], reverse=True)
        return topic_docs

    def top_words_topic(self, topic_id, num_words=10):
        """
        Find the top n words for the given topic.

        Args:
            topic_id: The topic from which we want the top words.
            num_words: The amount of words from that topic that we want.

        Returns:
            A list of tuples with the words and their similarity to the topic.
        """
        # Check the topic exists.
        if topic_id not in self.topic_words:
            raise NameError("Topic not found.")

        # Check we are not giving more words than what we can.
        word_count = min(num_words, len(self.topic_words[topic_id]))

        # The list of tuples with words and similarities.
        result = self.topic_words[topic_id][:word_count]
        return result

    def all_topics_top_words(self, num_words=10):
        """
        Make a list with the top words per topics.

        Returns: A list of top_words_topic(), that return a list of the top words
            per topic.
        """
        top_words = []
        for topic_id, _ in self.top_topics():
            top_words.append(self.top_words_topic(topic_id, num_words=num_words))
        # List of words per topic.
        return top_words

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
        total = len(self.corpus_cord_uids)

        # Go through the tokens of all the documents in the Corpus' Sample.
        words_embeddings = {}
        for document in self.corpus_sample.docs_full_texts():
            doc_tokens = doc_tokenizer(document)
            # Add the new words from the document.
            for doc_word in doc_tokens:
                if doc_word not in words_embeddings:
                    word_embed = self.doc_model.word2vec(doc_word)
                    # Ignore words that the model can't encode (Zero Values).
                    if not numpy.any(word_embed):
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
        total = len(self.corpus_cord_uids)

        docs_embeddings = {}
        for cord_uid in self.corpus_cord_uids:
            # Select the content of the paper depending on the model used.
            if self.model_type == 'glove':
                # No length restrictions, use all text.
                doc_content = self.corpus_manager.paper_full_text(cord_uid)
            elif self.model_type == 'bert':
                # BERT has a length limit, use only title and abstract.
                doc_content = self.corpus_manager.paper_title_abstract(cord_uid)
            elif self.model_type == 'specter':
                # SPECTER only uses the title and abstract.
                doc_content = self.corpus_manager.paper_title_abstract(cord_uid)
            else:
                raise NameError(f"We don't support <{self.model_type}> yet.")

            # Encode the Documents' Content.
            doc_embedding = self.doc_model.doc2vec(doc_content)
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
        # Get Document Embeddings
        doc_embeddings = list(self.doc_embeds.values())

        # Use UMAP to reduce the dimensions of the embeddings.
        umap_embeddings = umap.UMAP(n_neighbors=15,
                                    n_components=5,
                                    metric='cosine').fit_transform(doc_embeddings)

        # Use HDBSCAN to find the cluster of documents in the vector space.
        clusterer = hdbscan.HDBSCAN(min_cluster_size=15,
                                    metric='euclidean',
                                    cluster_selection_method='eom')
        cluster_labels = clusterer.fit(umap_embeddings)

        # Save the embeddings per topic label.
        topic_labels = cluster_labels.labels_
        topic_label_embeds = {}
        for label, doc_embed in zip(topic_labels, doc_embeddings):
            # Skip Noise labels:
            if label == -1:
                continue
            # Check if this the first time we find this topic.
            if label not in topic_label_embeds:
                topic_label_embeds[label] = [doc_embed]
            else:
                topic_label_embeds[label].append(doc_embed)

        # Find the average embed per label.
        topic_embeddings = {}
        for topic_label, label_embeds in topic_label_embeds.items():
            # # Transform embeddings from Torch to Numpy to find average.
            # numpy_embeds = [embed.numpy() for embed in label_embeds]
            # mean_embeds = numpy.mean(numpy_embeds, axis=0)
            # # Transform the average embedding into torch again.
            # topic_torch_embed = torch.from_numpy(mean_embeds)
            # # Save the topic embedding.
            # topic_embeddings[topic_label] = topic_torch_embed
            mean_embeds = numpy.mean(label_embeds, axis=0)
            topic_embeddings[topic_label] = mean_embeds

        return topic_embeddings

    def _find_doc_topics(self, show_progress=False):
        """
        Create a dictionary assigning each document to their closest topic.

        Returns: A dictionary with the topic ID as key, and the list of documents
            belonging to the topics as value.
        """
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
            tuple_word_sim.sort(key=lambda word_sim: word_sim[1], reverse=True)
        return topic_words

    def _closest_topic(self, embedding):
        """
        Given the embedding of a document or a word, find the closest topic to
        this embedding using cosine similarity.

        Args:
            embedding: A Tensor (most likely) with the vector of the word or
                document we want to classify.

        Returns:
            A tuple with the ID of the closest topic and their similarity.
        """
        # Use iterator to get the topic and their embeddings
        topic_iterator = iter(self.topic_embeds.items())

        # First topic cosine similarity.
        closest_topic, topic_embed = next(topic_iterator)
        max_similarity = float(util.cos_sim(embedding, topic_embed))

        # Go through the rest of the topics.
        for topic, topic_embed in topic_iterator:
            new_similarity = float(util.cos_sim(embedding, topic_embed))
            if new_similarity > max_similarity:
                # New closer topic.
                closest_topic = topic
                max_similarity = new_similarity

        # The closest topic ID.
        return closest_topic, max_similarity


if __name__ == '__main__':
    # Record the Runtime of the Program
    stopwatch = TimeKeeper()

    # Test TopicModel class.
    print("\nLoading Topic Model...")
    topic_model = TopicModel(doc_type='medium', sample_size=1_000,
                             model_type='glove', show_progress=True)
    print("Done.")
    print(f"[{stopwatch.formatted_runtime()}]")

    num_topics = topic_model.num_topics
    print(f"\n{num_topics} topics found.")

    print("\nTopics and Document count:")
    all_topics = topic_model.top_topics()
    for topic in all_topics:
        print(topic)

    top_n = 15
    print(f"\nTop {top_n} words per topic:")
    words_per_topic = topic_model.all_topics_top_words(top_n)
    for i, word_list in enumerate(words_per_topic):
        print(f"----> Topic <{i}>:")
        for word_sim in word_list:
            print(word_sim)

    print("\nDone.")
    print(f"[{stopwatch.formatted_runtime()}]\n")
