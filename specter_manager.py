# Gelin Eguinosa Rosique

import json
import numpy as np
from os import mkdir
from os.path import isdir, isfile, join
from collections import deque
# from random import sample

from document_model import DocumentModel
from papers import Papers
from cord19_utilities import load_vocab_embeddings, embeds_title_abstract
from extra_funcs import number_to_3digits, progress_bar
from time_keeper import TimeKeeper


class SpecterManager(DocumentModel):
    """
    Load & Manage the saved embeddings of the words and documents in the CORD-19
    dataset.
    """
    # Class Data Locations.
    data_folder = 'project_data'
    class_data_folder = 'specter_manager'
    papers_embeds_folder = 'papers_fragmented_embeds'
    vocab_embeds_folder = 'vocabulary_fragmented_embeds'
    papers_embeds_file = 'specter_papers_embeddings.json'
    vocab_embeds_file = 'specter_vocab_embeddings.json'
    papers_index_file = 'papers_fragment_embeds_index.json'
    vocab_index_file = 'vocab_fragment_embeds_index.json'
    papers_fragment_prefix = 'papers_fragment_'
    vocab_fragment_prefix = 'vocabulary_fragment_'

    # Attributes for Fragment Embedding's Dictionaries.
    paper_fragments_number = 500
    vocab_fragments_number = 500
    paper_cache_size = 50
    vocab_cache_size = 50

    def __init__(self, load_full_dicts=False, show_progress=False):
        """
        Load and save an instance of the class Papers() to get access to the
        papers embeddings in CORD-19, and also get the specter embeddings for
        the vocabulary of the content we are going to use.

        By default, if no 'cord19_papers' or 'vocab_embeddings' is provided,
        creates and loads its own Papers() class and the vocabulary embeddings
        for words in the Titles and Abstracts of the CORD-19 dataset.

        Args:
            load_full_dicts: Bool showing if we are going to load to memory
                the entire embedding's dictionary of the vocabulary and papers
                or use the fragmented versions of the dictionaries, so it is
                lighter on memory.
            show_progress: Bool representing whether we show the progress of
                the function or not.
        """
        # Save whether we'll load all the embeddings to memory or fragments of
        # the embeddings dictionaries.
        self.full_dicts = load_full_dicts

        # Check if the data folders were created.
        if not isdir(self.data_folder):
            mkdir(self.data_folder)
        class_folder_path = join(self.data_folder, self.class_data_folder)
        if not isdir(class_folder_path):
            mkdir(class_folder_path)
        papers_folder_path = join(class_folder_path, self.papers_embeds_folder)
        if not isdir(papers_folder_path):
            mkdir(papers_folder_path)
        vocab_folder_path = join(class_folder_path, self.vocab_embeds_folder)
        if not isdir(vocab_folder_path):
            mkdir(vocab_folder_path)

        if self._class_saved():
            # Load the type of dictionaries we are using (Full/Fragmented).
            if self.full_dicts:
                # Load full Dictionaries.
                if show_progress:
                    print("Loading all Paper' Embeddings to memory...")
                paper_embeds_path = join(class_folder_path, self.papers_embeds_file)
                with open(paper_embeds_path, 'r') as f:
                    paper_embeds = json.load(f)
                if show_progress:
                    print("Loading all the Vocabulary's Embeddings to memory...")
                vocab_embeds_path = join(class_folder_path, self.vocab_embeds_file)
                with open(vocab_embeds_path, 'r') as f:
                    vocab_embeds = json.load(f)
                # Dicts not Loaded.
                paper_fragments_index = None
                vocab_fragments_index = None
            else:
                # Dicts not Loaded.
                paper_embeds = None
                vocab_embeds = None
                # Load Fragment's Index Dictionaries.
                if show_progress:
                    print("Loading Index of Paper's Embeddings...")
                papers_index_path = join(class_folder_path, self.papers_index_file)
                with open(papers_index_path, 'r') as f:
                    paper_fragments_index = json.load(f)
                if show_progress:
                    print("Loading Index of Vocabulary's Embeddings...")
                vocab_index_path = join(class_folder_path, self.vocab_index_file)
                with open(vocab_index_path, 'r') as f:
                    vocab_fragments_index = json.load(f)
        else:
            # Load & Save Papers' Embeddings.
            if show_progress:
                print("Creating instance of Papers() class...")
            corpus = Papers(show_progress=show_progress)
            papers_ids = list(corpus.papers_cord_uids())
            # Create dictionary with all the Paper's embeddings.
            if show_progress:
                print("Loading paper's embeddings using Papers() class...")
            # Progress Variables.
            count = 0
            total = len(papers_ids)
            paper_embeds = {}
            for paper_id in papers_ids:
                paper_embeds[paper_id] = corpus.paper_embedding(paper_id)
                if show_progress:
                    count += 1
                    progress_bar(count, total)

            # Save full dictionary with Paper's embeddings.
            if show_progress:
                print("Saving Full Dictionary with the paper's embeddings...")
            paper_embeds_path = join(class_folder_path, self.papers_embeds_file)
            with open(paper_embeds_path, 'w') as f:
                json.dump(paper_embeds, f)
            # Save Paper's embeddings in fragments.
            if show_progress:
                print("Creating Index & Fragments of the paper's embeddings...")
            paper_fragments_index = create_embeddings_index(embeds_dict=paper_embeds,
                                                            folder_path=papers_folder_path,
                                                            file_prefix=self.papers_fragment_prefix,
                                                            fragments_num=self.paper_fragments_number,
                                                            show_progress=show_progress)
            # Save Paper's Fragments Index.
            papers_index_path = join(class_folder_path, self.papers_index_file)
            with open(papers_index_path, 'w') as f:
                json.dump(paper_fragments_index, f)

            # Load & Save Vocabulary Embeddings.
            if show_progress:
                print("Loading the vocabulary's embeddings...")
            vocab_embeds = load_vocab_embeddings(embeds_title_abstract)
            # Save Full Dictionary with Vocabulary Embeddings.
            if show_progress:
                print("Saving Full Dictionary of the vocabulary's embeddings...")
            vocab_embeds_path = join(class_folder_path, self.vocab_embeds_file)
            with open(vocab_embeds_path, 'w') as f:
                json.dump(vocab_embeds, f)
            # Save Vocabulary's Embeddings in Fragments.
            if show_progress:
                print("Saving Index & Fragments of the vocabulary's embeddings...")
            vocab_fragments_index = create_embeddings_index(embeds_dict=vocab_embeds,
                                                            folder_path=vocab_folder_path,
                                                            file_prefix=self.vocab_fragment_prefix,
                                                            fragments_num=self.vocab_fragments_number,
                                                            show_progress=show_progress)
            # Save Vocabulary's Fragments Index.
            vocab_index_path = join(class_folder_path, self.vocab_index_file)
            with open(vocab_index_path, 'w') as f:
                json.dump(vocab_fragments_index, f)

        # Save the type of dicts we are going to use (Full/Fragmented)
        if self.full_dicts:
            if show_progress:
                print("Using Full Embeddings' Dictionary.")
            self.full_paper_embeds = paper_embeds
            self.full_vocab_embeds = vocab_embeds
            self.paper_embeds_index = None
            self.vocab_embeds_index = None
            self.paper_cached_dicts = None
            self.vocab_cached_dicts = None
            self.paper_dict_queue = None
            self.vocab_dict_queue = None
        else:
            if show_progress:
                print("Using Fragments to load the Embeddings' Dictionary.")
            self.full_paper_embeds = None
            self.full_vocab_embeds = None
            self.paper_embeds_index = paper_fragments_index
            self.vocab_embeds_index = vocab_fragments_index
            self.paper_cached_dicts = {}
            self.vocab_cached_dicts = {}
            self.paper_dict_queue = deque()
            self.vocab_dict_queue = deque()

    def model_type(self):
        """
        Give the type of the document model that was used to create the
        embeddings.

        Returns: String with the name of the model 'specter'.
        """
        return 'specter'

    def word_vector(self, word):
        """
        Use the embeddings' dictionary to find the specter vector for the given
        'word'. If the word is not found in the vocabulary, then return a Zero
        vector.

        Args:
            word: String containing the word.

        Returns:
            Numpy.ndarray containing the embedding of the word.
        """
        # Check what type of dictionary we are using (Full/Fragmented).
        if self.full_dicts:
            word_embed = self.full_vocab_embeds.get(word, [0])
        else:
            # Using fragmented dicts.
            fragment_filename = self.vocab_embeds_index.get(word, None)
            # Word not in Vocabulary.
            if not fragment_filename:
                word_embed = [0]
            # Check if word embed dictionary is already cached.
            elif fragment_filename in self.vocab_cached_dicts:
                word_embed_dict = self.vocab_cached_dicts[fragment_filename]
                word_embed = word_embed_dict[word]
            else:
                # Load the Fragment Embed Dictionary of the Word.
                fragment_path = join(self.data_folder, self.class_data_folder,
                                     self.vocab_embeds_folder, fragment_filename)
                with open(fragment_path, 'r') as f:
                    fragment_dict = json.load(f)
                # Save Fragment in Cache.
                self.vocab_cached_dicts[fragment_filename] = fragment_dict
                self.vocab_dict_queue.append(fragment_filename)
                # Check if we have exceeded the cache size:
                if len(self.vocab_cached_dicts) > self.vocab_cache_size:
                    # Delete oldest cached dictionary.
                    oldest_fragment_name = self.vocab_dict_queue.popleft()
                    del self.vocab_cached_dicts[oldest_fragment_name]
                # Get Word's Embedding.
                word_embed = fragment_dict[word]

        # Transform embedding to Numpy.ndarray
        numpy_embed = np.array(word_embed)
        return numpy_embed

    def document_vector(self, cord_uid):
        """
        Use the CORD-19 dataset to find the specter embedding for the given
        'doc_cord_uid' paper.

        Args:
            cord_uid: String with the ID of the paper we want to find the
                embedding.

        Returns:
            Numpy.ndarray with the specter embedding of the document.
        """
        # Check the type of dictionary we are using (Full/Fragmented).
        if self.full_dicts:
            paper_embed = self.full_paper_embeds[cord_uid]
        else:
            # Using Fragmented dicts.
            fragment_filename = self.paper_embeds_index[cord_uid]
            # Check if the Fragment Dictionary is cached.
            if fragment_filename in self.paper_cached_dicts:
                paper_embed_dict = self.paper_cached_dicts[fragment_filename]
                paper_embed = paper_embed_dict[cord_uid]
            else:
                # Load the Fragment containing the Paper's Embedding.
                fragment_path = join(self.data_folder, self.class_data_folder,
                                     self.papers_embeds_folder, fragment_filename)
                with open(fragment_path, 'r') as f:
                    fragment_dict = json.load(f)
                # Save Fragment in cache.
                self.paper_cached_dicts[fragment_filename] = fragment_dict
                self.paper_dict_queue.append(fragment_filename)
                # Check if we have exceeded the cache size.
                if len(self.paper_cached_dicts) > self.paper_cache_size:
                    # Delete oldest cached dictionary.
                    oldest_fragment_name = self.paper_dict_queue.popleft()
                    del self.paper_cached_dicts[oldest_fragment_name]
                # Get Paper's Embedding.
                paper_embed = fragment_dict[cord_uid]

        numpy_embed = np.array(paper_embed)
        return numpy_embed

    def _class_saved(self):
        """
        Check the class' files to see if we can properly load the embeddings of
        the papers and vocabulary without having to use the Papers() class or
        the static method load_vocab_embeddings().

        Returns: Bool showing if all the local files of the class are saved.
        """
        # Check data folders.
        if not isdir(self.data_folder):
            return False
        class_folder_path = join(self.data_folder, self.class_data_folder)
        if not isdir(class_folder_path):
            return False
        papers_folder_path = join(class_folder_path, self.papers_embeds_folder)
        if not isdir(papers_folder_path):
            return False
        vocab_folder_path = join(class_folder_path, self.vocab_embeds_folder)
        if not isdir(vocab_folder_path):
            return False

        # Check class Files.
        papers_embeds_path = join(class_folder_path, self.papers_embeds_file)
        if not isfile(papers_embeds_path):
            return False
        vocab_embeds_path = join(class_folder_path, self.vocab_embeds_file)
        if not isfile(vocab_embeds_path):
            return False
        papers_index_path = join(class_folder_path, self.papers_index_file)
        if not isfile(papers_index_path):
            return False
        vocab_index_path = join(class_folder_path, self.vocab_index_file)
        if not isfile(vocab_index_path):
            return False

        # All files checked and ready.
        return True


def create_embeddings_index(embeds_dict: dict, folder_path: str, file_prefix: str,
                            fragments_num=500, show_progress=False):
    """
    Split an embeddings' dictionary into fragments and save the fragments inside
    'folder_path', also creating an index dictionary to know where the
    embeddings are stored.

    Args:
        embeds_dict: Dictionary containing all the embeddings we are going to
            save.
        folder_path: String with the path to the folder where we are going to
            store the fragments of the embeddings' dictionary.
        file_prefix: String with the prefix used to create the filenames of the
            fragments of the embeddings' dictionary.
        fragments_num: Int with the number of fragments we are going to create.
            The number of fragments must be between 0-999.
        show_progress: Bool representing whether we show the progress of
            the function or not.

    Returns:
        Dictionary containing the index for the embeddings and their locations
            in the fragments.
    """
    # Check we don't have more fragments and available embeddings.
    total_embeds = len(embeds_dict)
    if fragments_num > total_embeds:
        num_fragments = total_embeds
    else:
        num_fragments = fragments_num

    # Calculate amount embeddings per fragment.
    num_fragment_embeds = total_embeds // num_fragments + 1

    # Embeddings Dict & Temporary Fragment Dict variables.
    embeds_index = {}
    fragment_dict = {}
    created_fragments = 1
    fragment_filename = file_prefix + number_to_3digits(created_fragments) + '.json'

    # Progress Variables.
    count = 0
    total = total_embeds
    # Save Embeddings.
    for embed_id, embedding in embeds_dict.items():
        # Embedding Data.
        embeds_index[embed_id] = fragment_filename
        fragment_dict[embed_id] = embedding

        # Check if the fragment is full.
        if len(fragment_dict) >= num_fragment_embeds:
            # Save current fragment.
            fragment_path = join(folder_path, fragment_filename)
            with open(fragment_path, 'w') as f:
                json.dump(fragment_dict, f)
            # Reset Fragment variables.
            fragment_dict = {}
            created_fragments += 1
            fragment_filename = file_prefix + number_to_3digits(created_fragments) + '.json'

        # Progress.
        if show_progress:
            count += 1
            progress_bar(count, total)

    # Index containing the embedding IDs and their fragments' filenames.
    return embeds_index


if __name__ == '__main__':
    # Record Runtime of the Program.
    stopwatch = TimeKeeper()

    # Test Specter Manager.
    print("\nLoading Specter Manager...")
    the_manager = SpecterManager(show_progress=True)
    print("Done.")
    print(f"[{stopwatch.formatted_runtime()}]")

    # Get words embeddings:
    while True:
        input_word = input("\nType a word from the vocabulary: ")
        if input_word in {'', 'q', 'quit'}:
            break
        input_embed = the_manager.word_vector(input_word)
        print(f"Specter embedding of <{input_word}>:")
        print(input_embed)
        print(f"Type: {type(input_embed)}")

    # Get Paper's embeddings:
    while True:
        input_id = input("\nType the cord_uid of the paper: ")
        if input_id in {'', 'q', 'quit'}:
            break
        the_paper_embed = the_manager.document_vector(input_id)
        print(f"Specter embedding of paper <{input_id}>:")
        print(the_paper_embed)
        print(f"Type: {type(the_paper_embed)}")

    # # Testing cache size.
    # # Vocabulary's Embeddings.
    # iter_count = 0
    # vocab_list = sample(list(the_manager.vocab_embeds_index), 75)
    # for the_word in vocab_list:
    #     the_embed = the_manager.word_vector(the_word)
    #     print(f"Specter embedding of <{the_word}>:")
    #     print(the_embed)
    #     print(f"Type {type(the_embed)}")
    #     iter_count += 1
    #     print("Iteration:", iter_count)
    # # Paper's Embeddings.
    # iter_count = 0
    # paper_ids = sample(list(the_manager.paper_embeds_index), 75)
    # for the_id in paper_ids:
    #     the_paper_embed = the_manager.document_vector(the_id)
    #     print(f"Specter embedding of paper <{the_id}>:")
    #     print(the_paper_embed)
    #     print(f"Type: {type(the_paper_embed)}")
    #     iter_count += 1
    #     print("Iteration:", iter_count)

    print("Done.")
    print(f"[{stopwatch.formatted_runtime()}]\n")
