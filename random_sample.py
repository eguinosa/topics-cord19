# Gelin Eguinosa Rosique
# 2022

import json
import random
from os import mkdir, listdir
from os.path import join, isdir, isfile

from corpus_cord19 import CorpusCord19
from papers_analyzer import PapersAnalyzer
from extra_funcs import progress_bar, progress_msg, big_number
from time_keeper import TimeKeeper


class RandomSample(CorpusCord19):
    """
    Create and Manage a random sample of the CORD-19 papers, controlling the
    sample size and the type of papers we take (small, medium or big).
    """
    # Class Data Locations
    data_folder = 'project_data'
    random_sample_folder = 'random_sample_files'
    default_index_file = 'random_sample_default.json'
    sample_index_prefix = 'random_sample_'

    def __init__(self, paper_type='all', sample_size=-1, show_progress=False,
                 _use_saved=False, _saved_id=None):
        """
        Create a Random Sample of documents from the CORD-19 dataset, or load a
        previously created Random Sample.

        Args:
            paper_type: The type of Papers we want to use for the Sample (small,
                medium or big).
            sample_size: An int with size of the sample. The default value '-1'
                represents all the papers available with the specified paper
                type.
            _use_saved: A Bool indicating if we are loading the sample from a
                file.
            _saved_id: The ID of the sample we want to load. It can be used when
                we previously saved a sample with an ID, to avoid loading the
                last used sample.
            show_progress: A Bool representing whether we show the progress of
                the function or not.
        """
        # Check the project folders exist.
        if not isdir(self.data_folder):
            mkdir(self.data_folder)
        sample_folder_path = join(self.data_folder, self.random_sample_folder)
        if not isdir(sample_folder_path):
            mkdir(sample_folder_path)

        # Get the analyzed CORD-19 papers.
        self.analyzed_papers = PapersAnalyzer(show_progress=show_progress)
        # Also save the unorganized Papers().
        self.cord19_papers = self.analyzed_papers.cord19_papers

        # Use a saved Random Sample.
        if _use_saved:
            # Check if the sample index is locally available.
            if _saved_id:
                # Load a specific Sample, previously saved with an ID.
                saved_sample_file = self.sample_index_prefix + _saved_id + '.json'
                self.sample_index_path = join(sample_folder_path, saved_sample_file)
            else:
                # Load last used Sample.
                self.sample_index_path = join(sample_folder_path, self.default_index_file)
            if not isfile(self.sample_index_path):
                raise FileNotFoundError("No local RandomSample index file available.")
            # Load Sample Index.
            with open(self.sample_index_path, 'r') as f:
                self.sample_cord_uids = json.load(f)
            if show_progress:
                progress_msg("Loading Sample...")
                progress_bar(1, 1)
        else:
            # Create a new Random Sample.
            self.sample_cord_uids = self._new_random_sample(paper_type=paper_type, sample_size=sample_size)
            # Progress
            if show_progress:
                progress_msg("Creating new Random Sample...")
                progress_bar(1, 1)
            # Save the new Sample Index.
            self.sample_index_path = join(sample_folder_path, self.default_index_file)
            with open(self.sample_index_path, 'w') as f:
                json.dump(self.sample_cord_uids, f)

    def _new_random_sample(self, paper_type='all', sample_size=-1):
        """
        Create a random sample of papers from the CORD-19 dataset, with the
        specified paper type and sample size.

        Args:
            paper_type: A string with the type of papers we want in the sample.
                They can be 'small', 'medium', 'big' or 'all' types.
            sample_size: An int with size of the sample. The default value '-1'
                represents all the papers available with the specified paper
                type.
        Returns:
            A list with 'cord_uid' of all the papers included in the Sample.
        """
        # Get the specified type of papers for the sample.
        if paper_type == 'small':
            papers = list(self.analyzed_papers.small_papers)
        elif paper_type == 'medium':
            papers = list(self.analyzed_papers.medium_papers)
        elif paper_type == 'big':
            papers = list(self.analyzed_papers.big_papers)
        elif paper_type == 'all':
            papers = list(self.cord19_papers.papers_index)
        else:
            raise NameError("The type of papers is not specified.")

        # Get the Sample Size:
        if sample_size < 0:
            total = len(papers)
        else:
            total = min(sample_size, len(papers))

        # Create the Random Sample.
        random_sample = random.sample(papers, total)
        return random_sample

    def papers_cord_uids(self):
        """
        Get the cord_uids of the papers in the Random Sample.

        Returns: A List[str] with the identifiers of the papers.
        """
        return self.sample_cord_uids

    def paper_title(self, cord_uid):
        """
        Get the title of the paper with the given 'cord_uid'.

        Args:
            cord_uid: A string with the identifier of the paper.

        Returns:
            A string containing the title of the paper.
        """
        return self.cord19_papers.paper_title(cord_uid)

    def paper_abstract(self, cord_uid):
        """
        Get the abstract of the paper with the given 'cord_uid'.

        Args:
            cord_uid: A string with the identifier of the paper.

        Returns:
            A string containing the abstract of the paper.
        """
        return self.cord19_papers.paper_abstract(cord_uid)

    def paper_body_text(self, cord_uid):
        """
        Get text in the body of the given 'cord_uid' paper, which is the content
        of the paper excluding the title and abstract.

        Args:
            cord_uid: A string with the identifier of the paper.

        Returns:
            A string with the body text of the paper.
        """
        return self.cord19_papers.paper_body_text(cord_uid)

    def paper_embedding(self, cord_uid):
        """
        Get the Specter embedding of the given 'cord_uid' paper.

        Args:
            cord_uid: A string with the identifier of the paper.

        Returns:
            A List[float] containing the embedding of the paper.
        """
        return self.cord19_papers.paper_embedding(cord_uid)

    def paper_authors(self, cord_uid: str):
        """
        Create a List with the names of the authors of the 'cord_uid' paper.

        Args:
            cord_uid: String with the ID of the paper.

        Returns:
            List[Dict] with the authors of the Paper. The Author's Dict will be
                in the form {'first_name': ..., 'last_name': ...}.
        """
        return self.cord19_papers.paper_authors(cord_uid)

    def paper_publish_date(self, cord_uid: str):
        """
        Extract the Publication Date of the Paper 'cord_uid'.

        Args:
            cord_uid: String with the ID of the paper.

        Returns:
            Dictionary with the 'year', 'month' and 'day' of the publication
            date of the paper.
        """
        return self.cord19_papers.paper_publish_date(cord_uid)

    def save_sample(self, sample_id):
        """
        Saves the current Random Sample in a separated index file, to avoid
        overwriting the index when we use a different Sample. The saved Sample
        can be loaded later providing the 'sample_id'.

        Args:
            sample_id: A string or number as the Sample ID.
        """
        # Create Random Sample path.
        sample_index_file = self.sample_index_prefix + str(sample_id) + '.json'
        sample_folder_path = join(self.data_folder, self.random_sample_folder)
        sample_index_path = join(sample_folder_path, sample_index_file)
        # Save Sample Index in file.
        with open(sample_index_path, 'w') as f:
            json.dump(self.sample_cord_uids, f)

    @classmethod
    def load(cls, sample_id=None, show_progress=False):
        """
        Load a previously saved Random Sample.
        - If no 'sample_id' is provided, it loads the last used RandomSample.

        Args:
            sample_id: A string with the ID that the Random Sample was saved with.
            show_progress: A Bool representing whether we show the progress of
                the function or not.

        Returns: A RandomSample()
        """
        # Load RandomSample
        saved_sample = cls(_use_saved=True, _saved_id=sample_id,
                           show_progress=show_progress)
        return saved_sample

    @classmethod
    def saved_samples(cls):
        """
        Create a list containing the IDs of the saved Random Samples.

        Returns: List[string] with the saved samples' IDs.
        """
        # Check if the project folders exist.
        if not isdir(cls.data_folder):
            return []
        sample_folder_path = join(cls.data_folder, cls.random_sample_folder)
        if not isdir(sample_folder_path):
            return []

        # Find all the available IDs in the class folder.
        samples_ids = []
        for filename in listdir(sample_folder_path):
            # Filter out invalid files.
            if not filename.startswith(cls.sample_index_prefix):
                continue
            if not filename.endswith('.json'):
                continue
            prefix_len = len(cls.sample_index_prefix)
            new_id = filename[prefix_len:-5]
            # Valid ID name.
            samples_ids.append(new_id)

        # The IDs of the saved Samples.
        return samples_ids

    @classmethod
    def sample_saved(cls, sample_id=None):
        """
        Check if we can load a previously saved Random Sample, searching for the
        sample index in the project data folder.

        Args:
            sample_id: A string or number as the Sample ID.

        Returns:
            A bool representing if we can load the sample or not.
        """
        # Check if the project folders exist.
        if not isdir(cls.data_folder):
            return False
        sample_folder_path = join(cls.data_folder, cls.random_sample_folder)
        if not isdir(sample_folder_path):
            return False

        # Get the sample index path.
        if sample_id:
            # Check sample previously saved with an ID.
            saved_sample_file = cls.sample_index_prefix + str(sample_id) + '.json'
            sample_index_path = join(sample_folder_path, saved_sample_file)
        else:
            # Check last used Sample.
            sample_index_path = join(sample_folder_path, cls.default_index_file)

        # Check if the index file exists.
        result = isfile(sample_index_path)
        return result

    @classmethod
    def saved_sample_size(cls, sample_id=None):
        """
        Check the size of the 'sample_id' saved Sample. If no 'sample_id' is
        provided, checks the size of the last sample used.

        Args:
            sample_id:  A string or number as the Sample ID.

        Returns:
            A int with the number of papers available in the saved Sample. If no
                sample is found, then returns -1.
        """
        # Check if the Sample Index exist first.
        if not cls.sample_saved(sample_id):
            raise FileNotFoundError("There is no Random Sample available.")

        # Create Sample Path.
        sample_folder_path = join(cls.data_folder, cls.random_sample_folder)
        if sample_id:
            # Check sample previously saved with an ID.
            saved_sample_file = cls.sample_index_prefix + str(sample_id) + '.json'
            sample_index_path = join(sample_folder_path, saved_sample_file)
        else:
            # Check last used Sample.
            sample_index_path = join(sample_folder_path, cls.default_index_file)

        # Check the size of the index.
        with open(sample_index_path, 'r') as f:
            sample_index = json.load(f)
            sample_size = len(sample_index)

        return sample_size


if __name__ == '__main__':
    # Record the Runtime of the Program
    stopwatch = TimeKeeper()

    # # -- Loading & Saving with ID a used Random Sample. --
    # the_size = RandomSample.saved_sample_size()
    # print(f"\nSaved sample size: {big_number(the_size)}\n")
    #
    # print(f"\nLoading Saved Sample of {big_number(the_size)} papers.")
    # the_sample = RandomSample.load(show_progress=True)
    # print("Done.")
    # print(f"[{stopwatch.formatted_runtime()}]")
    #
    # new_id = str(the_size) + '_docs'
    # print(f"\nSaving the loaded Random Sample with ID <{new_id}>...")
    # the_sample.save_sample(sample_id=new_id)
    # print("Done.")
    # print(f"[{stopwatch.formatted_runtime()}]")

    # # -- Test the extraction of the content of the papers --
    # my_sample = RandomSample(paper_type='big', sample_size=5, show_progress=True)
    # # my_sample = RandomSample.load(show_progress=True)
    #
    # for paper_id in my_sample.papers_cord_uids():
    #     paper_content = my_sample.formatted_paper_content(paper_id)
    #     print(f"\nThe Content of the paper <{paper_id}>:")
    #     print("-----------------------------------------")
    #     max_length = 2_500
    #     if len(paper_content) >= max_length:
    #         paper_content = paper_content[:max_length] + '...[continues]...'
    #     print(paper_content)
    #     user_input = input("\ntype q/quit to exit otherwise continue: ")
    #     if user_input in {'q', 'quit', 'exit'}:
    #         break

    # -- Test the class. --
    test_size = 500
    print(f"\nCreating a Random Sample of {big_number(test_size)} documents...")
    sample = RandomSample('medium', test_size, show_progress=True)
    print("Done.")
    print(f"[{stopwatch.formatted_runtime()}]")

    print("\nFirst 5 Document of the sample:")
    count = 0
    for doc_id in sample.sample_cord_uids[:5]:
        count += 1
        print(f"Document {count} - cord_uid: {doc_id}")

    print(f"\nSaving Random Sample and creating a New One...")
    save_id = 'test_01'
    sample.save_sample(save_id)
    sample = RandomSample('medium', test_size)
    print("Done.")
    print(f"[{stopwatch.formatted_runtime()}]")

    print("\nLoading old Random Sample:...")
    old_sample = RandomSample.load(sample_id=save_id)
    print("Done.")
    print(f"[{stopwatch.formatted_runtime()}]")

    print("\nFirst 5 Document of the New Sample:")
    count = 0
    for doc_id in sample.sample_cord_uids[:5]:
        count += 1
        print(f"Document {count} - cord_uid: {doc_id}")

    print("\nFirst 5 Document of the Old Sample:")
    count = 0
    for doc_id in old_sample.sample_cord_uids[:5]:
        count += 1
        print(f"Document {count} - cord_uid: {doc_id}")

    print("\nDone.")
    print(f"[{stopwatch.formatted_runtime()}]\n")
