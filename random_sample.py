# Gelin Eguinosa Rosique

import json
from os.path import join, isdir, isfile
from random import sample

from papers_analyzer import PapersAnalyzer
from extra_funcs import progress_bar


class RandomSample:
    """
    Create and Manage a random sample of the CORD-19 papers, controlling the
    sample size and the type of papers we take (small, medium or big).
    """
    # Class Data Locations
    data_folder = 'project_data'
    sample_index_file = 'random_sample_index.json'

    def __init__(self, paper_type='all', size=-1, show_progress=False, _used_saved=False):
        """
        Create an instance of PapersAnalyzer() to get the papers' content and
        attributes from it.

        Args:
            show_progress: Bool representing whether we show the progress of
                the function or not.
            size: An int with size of the sample. The default value '-1'
                represents all the papers available with the specified paper
                type.
            show_progress: A Bool representing whether we show the progress of
                the function or not.
            _used_saved: A Bool indicating if we are loading the sample from a
                file. (Private Class Variable)
        """
        # Get the analyzed CORD-19 papers. (Creates data folder)
        self.analyzed_papers = PapersAnalyzer(show_progress=show_progress)
        # Also save the unorganized Papers().
        self.cord19_papers = self.analyzed_papers.cord19_papers

        # Use saved Random Sample.
        if _used_saved:
            # Check if the sample index is locally available.
            self.sample_index_path = join(self.data_folder, self.sample_index_file)
            if not isfile(self.sample_index_path):
                raise FileNotFoundError("No local RandomSample index file available.")
            # Load Sample Index.
            with open(self.sample_index_path, 'r') as f:
                self.sample_index = json.load(f)
            if show_progress:
                print("Loading Sample...")
                progress_bar(1, 1)
        else:
            # Create a new Random Sample.
            self.sample_index = self._new_random_sample(paper_type=paper_type,
                                                        size=size)
            # Progress
            if show_progress:
                print("Creating new Random Sample...")
                progress_bar(1, 1)
            # Save the new Sample Index.
            with open(self.sample_index_file, 'w') as f:
                json.dump(self.sample_index, f)

    def _new_random_sample(self, paper_type='all', size=-1):
        """
        Create a random sample of papers from the CORD-19 dataset, with the
        specified paper type and sample size.

        Args:
            paper_type: A string with the type of papers we want in the sample.
                They can be 'small', 'medium', 'big' or 'all' types.
            size: An int with size of the sample. The default value '-1'
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
        if size < 0:
            total = len(papers)
        else:
            total = min(size, len(papers))

        # Create the Random Sample.
        random_sample = sample(papers, total)
        return random_sample

    def sample_titles_abtracts(self):
        """
        Create an iterator with the titles and abstracts of the sample's papers.

        Returns: An iterator of strings.
        """
        for cord_uid in self.sample_index:
            yield self.cord19_papers.paper_title_abstract(cord_uid)

    def sample_contents(self):
        """
        Create an iterator containing the body text of the papers in the sample.

        Returns: An iterator of strings
        """
        for cord_uid in self.sample_index:
            yield self.cord19_papers.paper_content(cord_uid)

    def sample_full_texts(self):
        """
        Create and iterator containing the full text of the papers in the sample.

        Returns: An iterator of strings.
        """
        for cord_uid in self.sample_index:
            yield self.cord19_papers.paper_full_text(cord_uid)

    def sample_embeddings(self):
        """
        Create and iterator with the embeddings of all the papers in the sample.

        Returns: An iterator with the Specter vectors of the documents.
        """
        for cord_uid in self.sample_index:
            yield self.cord19_papers.paper_embedding(cord_uid)

    @classmethod
    def sample_saved(cls):
        """
        Check if we can load a previously saved Random Sample, searching for the
        sample index in the project data folder.

        Returns: A bool representing if we can load the sample or not.
        """
        # Check if the sample index file exists.
        sample_index_path = join(cls.data_folder, cls.sample_index_file)
        result = isdir(cls.data_folder) & isfile(sample_index_path)
        return result

    @classmethod
    def load_sample(cls, show_progress=False):
        """
        Load Random Sample from a saved sample index file.

        Args:
            show_progress: A Bool representing whether we show the progress of
                the function or not.
        Returns:
            A RandomSample() with the papers in the index file.
        """
        random_sample = cls(_used_saved=True, show_progress=show_progress)
        return random_sample
