# Gelin Eguinosa Rosique
# 2022

import json
from os import mkdir
from os.path import isdir, isfile, join

from papers_analyzer import PapersAnalyzer
from extra_funcs import progress_bar, progress_msg, big_number
from time_keeper import TimeKeeper


class ExtractCord19:
    """
    Extract the texts of the Papers in the CORD-19 dataset, classifying them by
    size. The texts will be saved in '.txt' files in 3 different folder
    depending on the size of the paper.
    - Small Papers: Containing one paragraph or less (0-300 characters).
    - Medium Papers: Containing one page or less (301-3,000).
    - Big Papers: Containing more than one page (3,001 or more).
    """
    # Project Data Locations
    project_data_folder = 'project_data'
    cord19_texts_folder = 'cord19_texts'
    small_papers_folder = 'small_papers'
    medium_papers_folder = 'medium_papers'
    big_papers_folder = 'big_papers'
    small_index_file = 'small_index.json'
    medium_index_file = 'medium_index.json'
    big_index_file = 'big_index.json'

    def __init__(self, show_progress=False):
        """
        Use the class PapersAnalyzer() to get the CORD-19 papers organized by
        size and save their texts, so they are more easily accessible.

        Args:
            show_progress: Bool representing whether we show the progress of
                the function or not.
        """
        # Create the file locations.
        self.texts_folder_path = join(self.project_data_folder, self.cord19_texts_folder)
        self.small_papers_path = join(self.texts_folder_path, self.small_papers_folder)
        self.medium_papers_path = join(self.texts_folder_path, self.medium_papers_folder)
        self.big_papers_path = join(self.texts_folder_path, self.big_papers_folder)
        self.small_index_path = join(self.texts_folder_path, self.small_index_file)
        self.medium_index_path = join(self.texts_folder_path, self.medium_index_file)
        self.big_index_path = join(self.texts_folder_path, self.big_index_file)

        # Check if the papers texts are already available.
        if self._papers_saved():
            # Load indexes.
            with open(self.small_index_path, 'r') as f:
                self.small_index = json.load(f)
            with open(self.medium_index_path, 'r') as f:
                self.medium_index = json.load(f)
            with open(self.big_index_path, 'r') as f:
                self.big_index = json.load(f)
            # Show progress.
            if show_progress:
                progress_msg("Loading Papers indexes...")
                progress_bar(3, 3)
        # Get papers' texts and create the texts' indexes.
        else:
            # Get Papers organized by size.
            papers = PapersAnalyzer(show_progress=show_progress)

            # Create Texts folder.
            if not isdir(self.texts_folder_path):
                mkdir(self.texts_folder_path)

            # Save texts for each of the papers' sizes.
            # Small Papers
            if show_progress:
                progress_msg("Saving Small Papers...")
            small_ids = list(papers.small_papers)
            small_texts = papers.cord19_papers.selected_papers_content(small_ids)
            self.small_index = _save_papers(self.small_papers_path,
                                            small_ids,
                                            small_texts,
                                            show_progress=show_progress)
            # Medium Papers
            if show_progress:
                progress_msg("Saving Medium Papers...")
            medium_ids = list(papers.medium_papers)
            medium_texts = papers.cord19_papers.selected_papers_content(medium_ids)
            self.medium_index = _save_papers(self.medium_papers_path,
                                             medium_ids,
                                             medium_texts,
                                             show_progress=show_progress)
            # Big Papers
            if show_progress:
                progress_msg("Saving Big Papers...")
            big_ids = list(papers.big_papers)
            big_texts = papers.cord19_papers.selected_papers_content(big_ids)
            self.big_index = _save_papers(self.big_papers_path,
                                          big_ids,
                                          big_texts,
                                          show_progress=show_progress)
            # Save Papers' indexes
            with open(self.small_index_path, 'w') as f:
                json.dump(self.small_index, f)
            with open(self.medium_index_path, 'w') as f:
                json.dump(self.medium_index, f)
            with open(self.big_index_path, 'w') as f:
                json.dump(self.big_index, f)

    def _papers_saved(self):
        """
        Check if the files containing the CORD-19 papers' texts were already
        created.

        Returns: A bool representing if the files are available or not.
        """
        # Check if the data_folder exists.
        if not isdir(self.project_data_folder):
            return False
        # Check if the Texts' folder exists.
        if not isdir(self.texts_folder_path):
            return False

        # Check the papers' folders exist.
        if not isdir(self.small_papers_path):
            return False
        if not isdir(self.medium_papers_path):
            return False
        if not isdir(self.big_papers_path):
            return False

        # Check if the indexes exist.
        if not isfile(self.small_index_path):
            return False
        if not isfile(self.medium_index_path):
            return False
        if not isfile(self.big_index_path):
            return False

        # All good.
        return True


def _save_papers(folder_path, docs_ids, docs_texts, show_progress=False):
    """
    Take the texts of a group of documents and save them in .txt files inside
    the provided folder.

    Args:
        folder_path: A String with the path of the folder were the documents'
            texts will be saved.
        docs_ids: An list of strings containing the ids of the documents in the
            same order as the documents texts.
        docs_texts: An iterable of strings containing the texts of the papers.
        show_progress: Bool representing whether we show the progress of
            the function or not.
    Returns:
        A dictionary with the name of the documents and the path of the files
            were they were saved.
    """
    # Create dictionary:
    docs_dict = {}

    # Create folder if it doesn't exist.
    if not isdir(folder_path):
        mkdir(folder_path)

    # Show progress variables.
    count = 0
    total = len(docs_ids)

    # Go through the documents ids and texts at the same time.
    for doc_id, doc_text in zip(docs_ids, docs_texts):
        # Create doc file path.
        doc_filename = doc_id + '.txt'
        doc_path = join(folder_path, doc_filename)
        # Save text to file.
        with open(doc_path, 'w') as f:
            f.write(doc_text)
        # Save the doc_id and filename in dictionary.
        docs_dict[doc_id] = doc_filename

        # Show progress.
        if show_progress:
            count += 1
            progress_bar(count, total)

    # Return the dictionary with the docs ids and filenames.
    return docs_dict


if __name__ == '__main__':
    # Record the runtime of the program.
    stopwatch = TimeKeeper()

    # Save the papers.
    print("\nLoading and Saving the CORD-19 papers...")
    extraction = ExtractCord19(show_progress=True)
    print("Done.")
    print(f"[{stopwatch.formatted_runtime()}]")

    small_count = len(extraction.small_index)
    medium_count = len(extraction.medium_index)
    big_count = len(extraction.big_index)
    total_count = small_count + medium_count + big_count
    print(f"\nSmall Papers saved: {big_number(small_count)}")
    print(f"Medium Papers saved: {big_number(medium_count)}")
    print(f"Big Papers saved: {big_number(big_count)}")
    print(f"Total: {big_number(total_count)}\n")
