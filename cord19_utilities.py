# Gelin Eguinosa Rosique

import json
from os import mkdir
from os.path import isdir, isfile, join

from corpus_cord19 import CorpusCord19
from document_model import DocumentModel
from papers import Papers
from specter_cord19 import SpecterCord19
from doc_tokenizers import doc_tokenizer
from random_sample import RandomSample
from time_keeper import TimeKeeper
from extra_funcs import progress_bar


# Function's Data Locations.
data_folder = 'project_data'
folder_titles_abstracts = 'cord19_titles_abstracts'

# Creating & Saving Vocabulary Locations.
vocabulary_folder = 'vocabulary_files'
vocab_default_file = 'vocabulary_temp.json'
vocab_titles_abstracts = 'vocabulary_titles_abstracts.json'
vocab_all_content = 'vocabulary_all_content.json'

# Vocabulary Embeddings Locations.
embeds_default_file = 'embeddings_vocabulary_temp.json'
embeds_title_abstract = 'embeddings_titles_abstracts.json'
embeds_all_content = 'embeddings_all_content.json'


def save_titles_abstracts(folder_name: str = None, corpus: CorpusCord19 = None,
                          show_progress=False):
    """
    Save the titles and abstracts of the given corpus to '.txt' files in a folder
    with the given 'folder_name'. The 'folder_name' will be located inside the
    'data_folder'.
        - By default, if no corpus is provided, creates a Corpus using the
          entire CORD-19 dataset.
        - If no folder name is provided, uses 'default_titles_abstracts'.

    Args:
        folder_name: A string with the name of the folder were the title and
            abstract of the papers will be stored.
        corpus: A class containing the documents selected for the corpus.
        show_progress: Bool representing whether we show the progress of
            the function or not.
    """
    # Check Project Data Folder exists.
    if not isdir(data_folder):
        mkdir(data_folder)

    # Create Corpus folder is it doesn't exist.
    if folder_name:
        folder_path = join(data_folder, folder_name)
    else:
        folder_path = join(data_folder, folder_titles_abstracts)
    if not isdir(folder_path):
        mkdir(folder_path)

    # Check the Corpus was provided.
    if not corpus:
        corpus = Papers()
    # Get Papers cord_uids.
    papers_ids = corpus.papers_cord_uids()

    # Progress Variables.
    count = 0
    total = len(papers_ids)
    # Iterate through the Papers' IDs.
    for cord_uid in papers_ids:
        # Load Title & Abstract.
        doc_title = corpus.paper_title(cord_uid)
        doc_abstract = corpus.paper_abstract(cord_uid)

        # Check they are not empty.
        if doc_title and doc_abstract:
            # Save the paper's title and abstract.
            title_abstract = doc_title + '\n\n' + doc_abstract
            file_name = cord_uid + '.txt'
            file_path = join(folder_path, file_name)
            with open(file_path, 'w') as f:
                f.write(title_abstract)

        # Show progress.
        if show_progress:
            count += 1
            progress_bar(count, total)


def save_corpus_vocabulary(corpus: CorpusCord19 = None, file_name: str = None,
                           use_title_abstract=False, show_progress=False):
    """
    Create a list containing all the words in the vocabulary of the
    provided 'corpus' and save it in 'file_name'.

    To create the vocabulary we can use the titles and abstracts of the corpus,
    or all the content of the documents, depending on the value of the
    'use_title_abstract' variable.

    By default, if no corpus is provided, it loads the entire CORD-19 dataset.

    Args:
        corpus: CorpusCord19 class containing the documents from where we are
            going to extract the vocabulary.
        file_name: String with the filename were we will store the vocabulary.
        use_title_abstract: Bool indicating whether we are using the title and
            abstract or the full content of the documents.
        show_progress: Bool representing whether we show the progress of
            the function or not.
    """
    # Load all papers in case no corpus was provided.
    if not corpus:
        corpus = Papers(show_progress=show_progress)

    # Create path for the file where we will store the vocabulary.
    if not isdir(data_folder):
        mkdir(data_folder)
    vocab_folder_path = join(data_folder, vocabulary_folder)
    if not isdir(vocab_folder_path):
        mkdir(vocab_folder_path)
    # Check if a filename was provided.
    if file_name:
        vocab_file_path = join(vocab_folder_path, file_name)
    else:
        vocab_file_path = join(vocab_folder_path, vocab_default_file)

    # Check if we are using title & abstract or all the content of the document.
    if use_title_abstract:
        content_provider = corpus.all_papers_title_abstract()
    else:
        content_provider = corpus.all_papers_content()

    # Progress variables.
    count = 0
    total = len(corpus.papers_cord_uids())
    # Create Vocabulary.
    vocab_words = set()
    for doc_content in content_provider:
        doc_tokens = doc_tokenizer(doc_content)
        vocab_words.update(doc_tokens)
        # Progress.
        if show_progress:
            count += 1
            progress_bar(count, total)

    # Save vocabulary.
    vocab_words = list(vocab_words)
    with open(vocab_file_path, 'w') as f:
        json.dump(vocab_words, f)


def load_corpus_vocabulary(vocab_file_name: str = None):
    """
    Load a saved corpus vocabulary. If no vocabulary filename is provided, loads
    the vocabulary from the default archive.

    Args:
        vocab_file_name: String with the filename were the vocabulary is
            stored.
    Returns:
        List[str] containing the saved vocabulary.
    """
    # Check the data folders exist.
    if not isdir(data_folder):
        raise FileNotFoundError("The is no data folder available to load file.")
    vocab_folder_path = join(data_folder, vocabulary_folder)
    if not isdir(vocab_folder_path):
        raise FileNotFoundError("There is no vocabulary folder available.")

    # Check if a filename was provided.
    if vocab_file_name:
        vocab_file_path = join(vocab_folder_path, vocab_file_name)
    else:
        vocab_file_path = join(vocab_folder_path, vocab_default_file)
    # Check if file exists.
    if not isfile(vocab_file_path):
        raise FileNotFoundError(f"There is no vocabulary file available.")

    # Load the file.
    with open(vocab_file_path, 'r') as f:
        corpus_vocab = json.load(f)
    # List of words with the corpus vocabulary.
    return corpus_vocab


def save_vocab_embeddings(vocab_list, model: DocumentModel = None,
                          file_name: str = None, show_progress=False):
    """
    Create and save dictionary containing the embeddings of all the words
    contained in the vocabulary 'vocab_list'.

    By default, if not Document Model is provided uses the Specter Model. If no
    filename is provided, uses a default filename.

    Saves a Dictionary with the String of the words as keys, and a List[float]
    with their embeddings as values.

    Args:
        vocab_list: List of strings containing the words in the vocabulary.
        model: DocumentModel class used to create the embeddings of the words.
        file_name: String with the filename were we will store the embeddings of
            the vocabulary.
        show_progress: Bool representing whether we show the progress of
                the function or not.
    """
    # Use the Specter Document Model by default.
    if not model:
        model = SpecterCord19()

    # Progress Variables.
    count = 0
    total = len(vocab_list)
    # Create the dictionary containing the embeddings.
    embeds_vocab = {}
    for vocab_word in vocab_list:
        # Get Numpy Embedding and transform it into a List[float].
        word_embed = model.word_vector(vocab_word)
        embed_list = word_embed.tolist()
        embeds_vocab[vocab_word] = embed_list

        # Show Progress.
        if show_progress:
            count += 1
            progress_bar(count, total)

    # Check if we need to create data folders.
    if not isdir(data_folder):
        mkdir(data_folder)
    vocab_folder_path = join(data_folder, vocabulary_folder)
    if not isdir(vocab_folder_path):
        mkdir(vocab_folder_path)
    # Create path for embeddings dict.
    if file_name:
        embeds_dict_path = join(vocab_folder_path, file_name)
    else:
        embeds_dict_path = join(vocab_folder_path, embeds_default_file)

    # Save dictionary using json.
    with open(embeds_dict_path, 'w') as f:
        json.dump(embeds_vocab, f)


def load_vocab_embeddings(dict_filename: str = None):
    """
    Load a saved dictionary containing the embeddings of the words belonging to
    a vocabulary.

    Args:
        dict_filename: The filename of the file were the dictionary is stored.

    Returns: Dictionary with the String of the words as keys, and a List[float]
        with their embeddings as values.
    """
    # Check the data folders exist.
    if not isdir(data_folder):
        raise FileNotFoundError("The is no data folder available to load file.")
    vocab_folder_path = join(data_folder, vocabulary_folder)
    if not isdir(vocab_folder_path):
        raise FileNotFoundError("There is no vocabulary folder available.")

    # Check if filename was provided.
    if dict_filename:
        embeds_dict_path = join(vocab_folder_path, dict_filename)
    else:
        embeds_dict_path = join(vocab_folder_path, embeds_default_file)
    # Check if the file exists.
    if not isfile(embeds_dict_path):
        raise FileNotFoundError("There is no dictionary with embeddings available.")

    # Load dictionary.
    with open(embeds_dict_path, 'r') as f:
        embedding_vocab = json.load(f)
    return embedding_vocab


if __name__ == '__main__':
    # Track Program Runtime.
    stopwatch = TimeKeeper()

    # Load Test Random Sample.
    doc_count = 5
    print(f"\nCreating Random Sample of {doc_count} medium documents.")
    # my_sample = RandomSample(paper_type='medium', sample_size=doc_count, show_progress=True)
    my_sample = RandomSample.load()
    print("Done.")
    print(f"[{stopwatch.formatted_runtime()}]")

    # Test save_corpus_vocabulary():
    print(f"\nSaving the Vocabulary of {doc_count} documents...")
    save_corpus_vocabulary(corpus=my_sample, use_title_abstract=False, show_progress=True)
    print("Done.")
    print(f"[{stopwatch.formatted_runtime()}]")

    # Test load_corpus_vocabulary():
    print(f"\nLoading the saved vocabulary...")
    the_vocabulary = load_corpus_vocabulary()
    print("Done.")
    print(f"[{stopwatch.formatted_runtime()}]")

    print("The vocabulary:")
    print(the_vocabulary)

    # Test creating and saving vocabulary embeddings.
    print("\nCreating & Saving embeddings for the vocabulary...")
    save_vocab_embeddings(the_vocabulary, show_progress=True)
    print("Done.")
    print(f"[{stopwatch.formatted_runtime()}]")

    # Test loading embeddings' dictionary.
    print("\nLoading embeddings' dictionary...")
    the_embed_dict = load_vocab_embeddings()
    print("Done.")
    print(f"[{stopwatch.formatted_runtime()}]")

    vocab_word = the_vocabulary[5]
    print(f"\nThe embedding of the word <{vocab_word}>:")
    print(the_embed_dict[vocab_word])

    # # Test save_titles_abstracts():
    # print("\nSaving Documents Titles and Abstracts to files...")
    # save_titles_abstracts(corpus=my_sample, show_progress=True)
    # print("Done.")
    # print(f"[{stopwatch.formatted_runtime()}]")

    # # -----------------------------------------------------------
    # # Save the Title & Abstract of all the papers in the CORD-19.
    # # -----------------------------------------------------------
    # print("\nLoading the CORD-19 Dataset...")
    # corpus_papers = Papers()
    # print("Done.")
    # print(f"[{stopwatch.formatted_runtime()}]")
    #
    # print("\nSaving Documents Titles and Abstracts to files...")
    # save_titles_abstracts(corpus=corpus_papers, show_progress=True)
    # print("Done.")
    # print(f"[{stopwatch.formatted_runtime()}]")

    # # -----------------------------------------------------------
    # # Save the vocabulary of the CORD-19 dataset.
    # # -----------------------------------------------------------
    # print("\nLoading CORD-19 dataset...")
    # corpus_papers = Papers(show_progress=True)
    # print("Done.")
    # print(f"[{stopwatch.formatted_runtime()}]")
    #
    # # Title & Abstract Vocabulary.
    # print(f"\nSaving the vocabulary of titles & abstracts in CORD-19...")
    # save_corpus_vocabulary(corpus=corpus_papers, file_name=vocab_titles_abstracts,
    #                        use_title_abstract=True, show_progress=True)
    # print("Done.")
    # print(f"[{stopwatch.formatted_runtime()}]")
    #
    # # All Content Vocabulary.
    # print(f"\nSave the vocabulary for all the content in the papers of CORD-19...")
    # save_corpus_vocabulary(corpus=corpus_papers, file_name=vocab_all_content,
    #                        use_title_abstract=False, show_progress=True)
    # print("Done.")
    # print(f"[{stopwatch.formatted_runtime()}]")
