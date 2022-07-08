# Gelin Eguinosa Rosique

import json
import numpy as np
from os.path import isdir, isfile, join

from extra_funcs import progress_bar
from time_keeper import TimeKeeper


class SimpleTopicModel:
    """
    Load a basic version of a saved Topic Model to access Topic's Attributes
    without having to load into memory the Vocabulary and Document's embeddings.
    Making the Topic Model much lighter on memory.
    """
    # Class Data Locations.
    data_folder = 'project_data'
    class_data_folder = 'topic_models'
    model_folder_prefix = 'topics_'
    default_model_id = 'default'
    model_index_file = 'topic_model_index.json'
    topic_embeds_file = 'topic_model_topic_embeds.json'

    def __init__(self, model_id: str = None, show_progress=False):
        """
        Load the main attributes of the saved Topic Model 'model_id'.

        Args:
            model_id: A string with the ID of a previously saved Topic Model.
            show_progress: Bool representing whether we show the progress of
                the function or not.
        """
        # Save Model's ID.
        if model_id:
            self.model_id = model_id
        else:
            self.model_id = self.default_model_id

        # Check the Existence of the Data Folders.
        if not isdir(self.data_folder):
            raise FileNotFoundError("There is no Data Folder available.")
        class_folder_path = join(self.data_folder, self.class_data_folder)
        if not isdir(class_folder_path):
            raise FileNotFoundError("There is no Class Data Folder available.")
        model_folder_name = self.model_folder_prefix + self.model_id
        model_folder_path = join(class_folder_path, model_folder_name)
        if not isdir(model_folder_path):
            raise FileNotFoundError(f"The Topic Model <{self.model_id}> has no Data Folder.")

        # Load the Main Indexes.
        if show_progress:
            print("Loading Topic Model's Index...")
        model_index_path = join(model_folder_path, self.model_index_file)
        if not isfile(model_index_path):
            raise FileNotFoundError(f"The Topic Model <{self.model_id}> has no Index available.")
        with open(model_index_path, 'r') as f:
            model_index = json.load(f)
        # Get Index Attributes.
        self.model_type = model_index['model_type']
        self.num_topics = model_index['num_topics']
        self.corpus_ids = model_index['corpus_ids']
        self.topic_docs = model_index['topic_docs']
        self.topic_words = model_index['topic_words']
        # Transform the Topic's IDs back into string (JSON save them as string).
        self.topic_docs = dict([(int(key), doc_ids)
                                for key, doc_ids in self.topic_docs.items()])
        self.topic_words = dict([(int(key), words)
                                 for key, words in self.topic_words.items()])

        # Load the Topic's Embeddings.
        if show_progress:
            print("Loading Topic's Embeddings...")
        topic_index_path = join(model_folder_path, self.topic_embeds_file)
        if not isfile(topic_index_path):
            raise FileNotFoundError(f"The Topic Model <{self.model_id}> has no"
                                    f" Topic's Embeddings available.")
        with open(topic_index_path, 'r') as f:
            topic_embeds_index = json.load(f)
        # Transform Topic's IDs to Int and Embeddings to Numpy.ndarray
        self.topic_embeds = {}
        # Progress Variables.
        count = 0
        total = len(topic_embeds_index)
        for topic_id, topic_embed in topic_embeds_index.items():
            self.topic_embeds[int(topic_id)] = np.array(topic_embed)
            if show_progress:
                count += 1
                progress_bar(count, total)

    def top_topics(self):
        """
        Make a sorted list with the topics organized by the amount of documents
        they represent.

        Returns:
            A list of tuples with the topics' ID and their document count.
        """
        # Create list of topics with their size.
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

        Args:
            num_words: The amount of words we want from each topic.

        Returns:
            A list of tuples with the Topic ID and their top_words_topic(),
                the latter containing the top words for the corresponding topic.
        """
        # Create list of words per topic.
        topics_top_words = []
        for topic_id, _ in self.top_topics():
            topic_top_words = self.top_words_topic(topic_id, num_words=num_words)
            topics_top_words.append((topic_id, topic_top_words))

        # Per topic, a list with the words and their similarity to the topic.
        return topics_top_words


if __name__ == '__main__':
    # Record Runtime of the Program.
    stopwatch = TimeKeeper()

    # Upload a Saved Model.
    model_name = 'test_bert_3000_docs'
    print(f"\nLoading Topic Model <{model_name}>...")
    the_model = SimpleTopicModel(model_id=model_name, show_progress=True)
    print("Done.")
    print(f"[{stopwatch.formatted_runtime()}]\n")

    print(f"\n{the_model.num_topics} topics found.")

    print("\nTopics and Document count:")
    all_topics = the_model.top_topics()
    for topic_and_size in all_topics:
        print(topic_and_size)

    

    print("Done.")
    print(f"[{stopwatch.formatted_runtime()}]\n")
