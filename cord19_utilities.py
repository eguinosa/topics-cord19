# Gelin Eguinosa Rosique
from os import mkdir
from os.path import isdir, join

from corpus_cord19 import CorpusCord19
from papers import Papers
from random_sample import RandomSample
from time_keeper import TimeKeeper
from extra_funcs import progress_bar

# Function's Data Locations
data_folder = 'project_data'
default_titles_abstracts = 'cord19_titles_abstracts'


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
        folder_path = join(data_folder, default_titles_abstracts)
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
        # Save the paper's title and abstract.
        title_abstract = corpus.paper_title_abstract(cord_uid)
        file_name = cord_uid + '.txt'
        file_path = join(folder_path, file_name)
        with open(file_path, 'w') as f:
            f.write(title_abstract)

        # Show progress.
        if show_progress:
            count += 1
            progress_bar(count, total)


if __name__ == '__main__':
    # Track Program Runtime.
    stopwatch = TimeKeeper()

    # Test save_titles_abstracts():
    doc_count = 30
    print(f"\nCreating Random Sample of {doc_count} medium documents.")
    my_sample = RandomSample(paper_type='big', sample_size=doc_count,
                             show_progress=True)
    print("Done.")
    print(f"[{stopwatch.formatted_runtime()}]")

    print("\nSaving Documents Titles and Abstracts to files...")
    save_titles_abstracts(corpus=my_sample, show_progress=True)
    print("Done.")
    print(f"[{stopwatch.formatted_runtime()}]")
