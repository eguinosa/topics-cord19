# Gelin Eguinosa Rosique
# 2022

import csv
import json
from os import mkdir
from os.path import join, isfile, isdir
from collections import defaultdict

from corpus_cord19 import CorpusCord19
from extra_funcs import progress_bar, progress_msg, number_to_3digits

# Testing Imports.
from random import choice
from pprint import pprint
from time_keeper import TimeKeeper
from extra_funcs import big_number


class PapersCord19(CorpusCord19):
    """
    Scans the CORD-19 dataset to create an index of it, saving all the relevant
    information for later use.
    """
    # CORD-19 Data Location
    cord19_data_folder = 'cord19_data'
    default_dataset = '2020-05-31'
    metadata_file = 'metadata.csv'
    embeddings_file = 'cord_19_embeddings_2020-05-31.csv'

    # Project Data Location
    project_data_folder = 'project_data'
    papers_data_folder = 'papers_data'
    project_embeds_folder = 'papers_embedding_dicts'
    papers_index_file = 'papers_index.json'
    embeds_index_file = 'papers_embeddings_index.json'

    def __init__(self, dataset_id='', show_progress=False):
        """
        Load the metadata.csv to create the index of all the papers available in
        the current CORD-19 dataset and save all the information of interest.

        Also, load the cord_19_embeddings to create an index and save them in
        a group of dictionaries, so they are quickly loaded without occupying
        too much memory.

        Args:
            dataset_id: String with the ID of the dataset we want to load.
            show_progress: Bool representing whether we show the progress of
                the function or not.
        """
        # Check if a Dataset ID was provided.
        if dataset_id:
            self.current_dataset = dataset_id
        else:
            self.current_dataset = self.default_dataset

        # Create the data folders if they don't exist.
        if not isdir(self.project_data_folder):
            mkdir(self.project_data_folder)
        class_data_path = join(self.project_data_folder, self.papers_data_folder)
        if not isdir(class_data_path):
            mkdir(class_data_path)

        # Form the papers index path.
        papers_index_path = join(class_data_path, self.papers_index_file)
        # Check if the papers' index exists or not.
        if isfile(papers_index_path):
            # Load the Papers' Index.
            with open(papers_index_path, 'r') as file:
                self.papers_index = json.load(file)
            # Papers Content Progress.
            if show_progress:
                progress_msg("Loading index of the papers' texts...")
                total = len(self.papers_index)
                progress_bar(total, total)
        else:
            # Announce what we are doing.
            if show_progress:
                progress_msg("Creating index of the papers' texts...")
            # Create the index of the papers.
            self.papers_index = self._create_papers_index(show_progress=show_progress)
            # Save the Papers' Index
            with open(papers_index_path, 'w') as file:
                json.dump(self.papers_index, file)

        # Create the folder of the embedding dictionaries if it doesn't exist.
        proj_embeds_folder_path = join(class_data_path, self.project_embeds_folder)
        if not isdir(proj_embeds_folder_path):
            mkdir(proj_embeds_folder_path)

        # Form the embeddings index path.
        embeds_index_path = join(class_data_path, self.embeds_index_file)
        # Check if the embeddings' index exists or not.
        if isfile(embeds_index_path):
            # Load the embeddings' index.
            with open(embeds_index_path, 'r') as file:
                self.embeds_index = json.load(file)
            # Papers Embeddings Progress.
            if show_progress:
                progress_msg("Loading index of papers' embeddings...")
                total = len(self.embeds_index)
                progress_bar(total, total)
        else:
            # Announce what we are doing.
            if show_progress:
                progress_msg("Creating index of the papers' embeddings...")
            # Save the embeddings of the papers and create an index of their
            # location.
            self.embeds_index = self._create_embeddings_index(embed_dicts=500, show_progress=show_progress)
            # Save the embeddings' index
            with open(embeds_index_path, 'w') as file:
                json.dump(self.embeds_index, file)

        # Create a Cache for the Embedding Dictionaries, to work faster in
        # repetitive cases.
        self.cached_embed_dict = {}
        self.cached_dict_filename = ''

    def _create_papers_index(self, show_progress=False):
        """
        Create an index of the papers available in the CORD-19 dataset specified
        in the data folders of the class.

        Args:
            show_progress: Bool representing whether we show the progress of
                the function or not.

        Returns:
            A dictionary containing the 'cord_uid' and the important data of all
                the papers in the CORD-19 dataset.
        """
        # Create the metadata path
        metadata_path = join(self.cord19_data_folder, self.current_dataset, self.metadata_file)

        # Dictionary where the information of the papers will be saved.
        papers_index = defaultdict(dict)

        # Initialize progress information:
        count = 0
        total = -1
        if show_progress:
            with open(metadata_path, 'r') as file:
                total = sum(1 for _ in file) - 1

        # Open the metadata file
        with open(metadata_path, 'r') as file:
            reader = csv.DictReader(file)
            # Go through the information of all the papers.
            for row in reader:
                # Get the fields of interest.
                cord_uid = row['cord_uid']
                title = row['title']
                abstract = row['abstract']
                publish_time = row['publish_time']
                authors = row['authors'].split('; ')
                pdf_json_files = row['pdf_json_files'].split('; ')
                pmc_json_files = row['pmc_json_files'].split('; ')

                # Save all the information of the current paper, or update it
                # if we have found this 'cord_uid' before. Also, check if they
                # are not empty.
                current_paper = papers_index[cord_uid]
                current_paper['cord_uid'] = cord_uid
                current_paper['title'] = title
                current_paper['abstract'] = abstract
                current_paper['publish_time'] = publish_time
                current_paper['authors'] = authors
                if pdf_json_files != ['']:
                    current_paper['pdf_json_files'] = pdf_json_files
                if pmc_json_files != ['']:
                    current_paper['pmc_json_files'] = pmc_json_files

                # Show progress if requested.
                if show_progress:
                    count += 1
                    progress_bar(count, total)

        # Transform the papers' index from a DefaultDict() to a normal dictionary.
        papers_index = dict(papers_index) 
        return papers_index

    def _create_embeddings_index(self, embed_dicts=100, show_progress=False):
        """
        Load all the embeddings of the documents from the current CORD-19
        dataset and save them in several dictionaries so when needed they can
        be loaded quickly and without occupying too much space in memory.

        We create an index with the 'cord_uid' of the papers and the location of
        the dictionary that contains them.

        The maximum accepted value for 'embed_dicts' is 999.

        Args:
            embed_dicts: The number of dictionaries that we are going to use
                to store all the embeddings of the papers (default: 100).
            show_progress: Bool representing whether we show the progress of
                the function or not.

        Returns:
            A dictionary (the index) containing the location of the dictionary
                that contains the embedding for a given paper.
        """
        # Get the amount of CORD-19 papers in the current dataset.
        total_papers = len(self.papers_index)
        # Amount of embeddings to be stored per dictionary
        embeds_per_dict = total_papers // embed_dicts + 1

        # Index to store the papers' 'cord_uid' and the location of the dictionary
        # with their embedding.
        embeddings_index = {}

        # Create a temporary dictionary to store the embeddings of the papers.
        temp_embeds_dict = {}
        # Create counter for the amount of dictionaries we have created to store
        # the embeddings of the papers, starting with 1.
        dicts_count = 1
        # Create counter for the amount of embeddings we have stored in the
        # current temporary dictionary.
        dict_embeds_count = 0
        # Create the name of the file where the temporary dictionary will be
        # stored.
        temp_dict_file = f"embeddings_dict_{number_to_3digits(dicts_count)}.json"

        # Create the path for the CSV file containing the embeddings.
        embeddings_path = join(self.cord19_data_folder, self.current_dataset, self.embeddings_file)

        # Initialize Progress Information.
        count = 0
        total = len(self.papers_index)

        # Load the file
        with open(embeddings_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            # Iterate through the embedding of all the papers
            for row in csv_reader:
                # Get the 'cord_uid' and check if we have seen it before.
                paper_cord_uid = row[0]
                if paper_cord_uid in embeddings_index:
                    continue
                # Get the embedding of the paper.
                paper_embedding = list(map(float, row[1:]))
                # Save the embedding in the temporary dictionary.
                temp_embeds_dict[paper_cord_uid] = paper_embedding
                # Save the file where the embedding of the current paper will be
                # saved.
                embeddings_index[paper_cord_uid] = temp_dict_file
                # Update the amount of embeddings stored.
                dict_embeds_count += 1

                # Check if we have reached the maximum amount of papers per dict.
                if dict_embeds_count >= embeds_per_dict:
                    # Save the current temporary dictionary
                    dict_folder_path = join(self.project_data_folder,
                                            self.papers_data_folder,
                                            self.project_embeds_folder)
                    temp_dict_path = join(dict_folder_path, temp_dict_file)
                    with open(temp_dict_path, 'w') as file:
                        json.dump(temp_embeds_dict, file)

                    # Reset the temporary dictionary.
                    temp_embeds_dict = {}
                    dicts_count += 1
                    dict_embeds_count = 0
                    temp_dict_file = f"embeddings_dict_{number_to_3digits(dicts_count)}.json"

                # Show progress if requested.
                if show_progress:
                    count += 1
                    progress_bar(count, total)

        # Check if we have pending embeddings once we finish visiting all the papers in
        # the CSV file.
        if temp_embeds_dict:
            # Save the current temporary dictionary
            dict_folder_path = join(self.project_data_folder,
                                    self.papers_data_folder,
                                    self.project_embeds_folder)
            temp_dict_path = join(dict_folder_path, temp_dict_file)
            with open(temp_dict_path, 'w') as file:
                json.dump(temp_embeds_dict, file)

        # Once we have saved all the embeddings, return the index.
        return embeddings_index

    def papers_cord_uids(self):
        """
        Create a list with the 'cord_uids' of the papers in the CORD-19 dataset.

        Returns: A List[str] with identifiers (cord_uids) of the papers.
        """
        # Take the cord_uids from the dictionary of the papers.
        all_cord_uids = list(self.papers_index)
        return all_cord_uids

    def paper_title(self, cord_uid: str):
        """
        Find the title of the CORD-19 paper specified by the 'cord_uid'
        identifier, and return them together as a string.

        Args:
            cord_uid: The Unique Identifier of the CORD-19 paper.

        Returns:
            A string containing the title of the paper.
        """
        # Use dictionary of the paper.
        paper_dict = self.papers_index[cord_uid]
        title_text = paper_dict['title']
        # Remove empty space at the beginning and end of the text.
        title_text = title_text.strip()
        return title_text

    def paper_abstract(self, cord_uid: str):
        """
        Find the title and abstract of the CORD-19 paper specified by the
        'cord_uid' identifier, and return them together as a string.

        Args:
            cord_uid: The Unique Identifier of the CORD-19 paper.

        Returns:
            A string containing the title and abstract of the paper.
        """
        # Use dictionary of the paper.
        paper_dict = self.papers_index[cord_uid]
        abstract_text = paper_dict['abstract']
        # Remove empty space at the beginning and end of the text.
        abstract_text = abstract_text.strip()
        return abstract_text

    def paper_body_text(self, cord_uid: str):
        """
        Find the text of the 'cord_uid' paper on either the 'pmc_json_files' or
        the 'pdf_json_files'.

        Args:
            cord_uid: The Unique Identifier of the CORD-19 paper.

        Returns:
            A string with the content of the paper, excluding the title and
                abstract.
        """
        # Get the dictionary with the info of the paper
        paper_dict = self.papers_index[cord_uid]
        # Get the paths for the documents of the paper
        doc_json_files = []
        if 'pmc_json_files' in paper_dict:
            doc_json_files += paper_dict['pmc_json_files']
        if 'pdf_json_files' in paper_dict:
            doc_json_files += paper_dict['pdf_json_files']

        # Load the content of all the paper's files, empty string as default.
        body_texts = ['']
        # Access the files and extract the text.
        for doc_json_file in doc_json_files:
            doc_json_path = join(self.cord19_data_folder, self.current_dataset, doc_json_file)
            new_body_text = paper_json_body_text(doc_json_path)
            body_texts.append(new_body_text)

        # Get the longest body text found for the paper.
        final_text = select_body_text(body_texts)
        final_text = final_text.strip()
        return final_text

    def paper_embedding(self, cord_uid: str):
        """
        Find the precomputed SPECTER Document Embedding for the specified Paper
        'cord_uid'.

        Args:
            cord_uid: The Unique Identifier of the CORD-19 paper.

        Returns:
            A 768-dimensional document embedding.
        """
        # Check if the 'cord_uid' is valid.
        if cord_uid not in self.embeds_index:
            raise NameError(f"The provided cord_uid <{cord_uid}> is not valid.")

        # Get the dictionary where the embedding is saved.
        embed_dict_filename = self.embeds_index[cord_uid]
        # Check if this dictionary is already in memory.
        if embed_dict_filename != self.cached_dict_filename:
            # If the needed dict is not in memory, load it.
            dict_folder_path = join(self.project_data_folder,
                                    self.papers_data_folder,
                                    self.project_embeds_folder)
            embed_dict_path = join(dict_folder_path, embed_dict_filename)
            with open(embed_dict_path, 'r') as file:
                # Update the cached dictionary.
                self.cached_embed_dict = json.load(file)
                self.cached_dict_filename = embed_dict_filename
        # Return the embedding using the cached dictionary.
        return self.cached_embed_dict[cord_uid]

    def paper_authors(self, cord_uid: str):
        """
        Create a List with the names of the authors of the 'cord_uid' paper.

        Args:
            cord_uid: String with the ID of the paper.

        Returns:
            List[Dict] with the authors of the Paper. The Author's Dict will be
                in the form {'first_name': ..., 'last_name': ...}.
        """
        # Get the authors for the papers.
        paper_info = self.papers_index[cord_uid]
        authors_list = paper_info['authors']

        # Create list for the formatted authors.
        formatted_authors = []
        for author_str in authors_list:
            # Skip when no author data provided.
            if not author_str:
                continue
            # 1 or more than 3 author data fields provided, only save the first.
            author_names = author_str.split(',')
            if len(author_names) == 1 or len(author_names) > 3:
                author_dict = {
                    'one_name': author_names[0].strip()
                }
            elif len(author_names) == 2:
                author_dict = {
                    'last_name': author_names[0].strip(),
                    'first_name': author_names[1].strip(),
                }
            elif len(author_names) == 3:
                author_dict = {
                    'last_name': author_names[0].strip(),
                    'middle_name': author_names[1].strip(),
                    'first_name': author_names[2].strip(),
                }
            else:
                raise Exception(f"Not Supported Version of Names for <{cord_uid}>.")
            # Add new author to the list.
            formatted_authors.append(author_dict)

        # List of Dictionaries with the authors.
        return formatted_authors

    def paper_publish_date(self, cord_uid: str):
        """
        Extract the Publication Date of the Paper 'cord_uid'.

        Args:
            cord_uid: String with the ID of the paper.

        Returns:
            Dictionary with the 'year', 'month' and 'day' of the publication
            date of the paper.
        """
        # Get the String with the Date of Publication.
        paper_info = self.papers_index[cord_uid]
        date_string = paper_info['publish_time']

        # Check we don't have an empty string.
        if not date_string:
            return {}

        # Separate Data Fields.
        date_split = date_string.split('-')

        # Create Dictionary with the Date Info.
        try:
            date_dict = {'year': int(date_split[0])}
            if len(date_split) > 1:
                date_dict['month'] = int(date_split[1])
            if len(date_split) > 2:
                date_dict['day'] = int(date_split[2])
        except ValueError:
            raise Exception(f"There is a Problem with the date of <{cord_uid}>.")
        # Dictionary with the publication date.
        return date_dict

    def selected_papers_title_abstract(self, cord_uids):
        """
        Create an iterator with the title and abstracts of the requested
        'cord_uids' papers.

        Args:
            cord_uids: A list of strings with the identifiers of the papers.

        Returns:
            An iterator of strings.
        """
        for cord_uid in cord_uids:
            yield self.paper_title_abstract(cord_uid)

    def selected_papers_body_text(self, cord_uids):
        """
        Create an iterator containing the body text of the papers requested in
        'cord_uids'.
        Args:
            cord_uids: A list of strings with the identifiers of the papers.

        Returns:
            An iterator of strings.
        """
        for cord_uid in cord_uids:
            yield self.paper_body_text(cord_uid)

    def selected_papers_content(self, cord_uids):
        """
        Create and iterator containing the full text of the papers requested in
        'cord_uids'.

        Args:
            cord_uids: A list of strings with the identifiers of the papers.

        Returns:
            An iterator of strings.
        """
        for cord_uid in cord_uids:
            yield self.paper_content(cord_uid)

    def selected_papers_embedding(self, cord_uids):
        """
        Create and iterator with the embeddings of all the papers requested in
        'cord_uids'.
        Args:
            cord_uids: A list of strings with the identifiers of the papers.

        Returns:
            An iterator with the Specter vectors of the documents.
        """
        for cord_uid in cord_uids:
            yield self.paper_embedding(cord_uid)


def paper_json_body_text(json_file_path: str):
    """
    Extract the body text of a paper from its CORD-19 json file.

    Args:
        json_file_path: A string with the path to the PMC or PDF json file.

    Returns:
        A string with the content in the body text of the paper.
    """
    # Where we are going to store the text of the paper.
    paper_body_text = ''

    # Open file and extract the dictionary with the content of the paper.
    with open(json_file_path, 'r') as f_json:
        # Get the dictionary containing all the info of the document.
        full_text_dict = json.load(f_json)

        # Get all the sections in the body of the document.
        last_section = ''
        for paragraph_dict in full_text_dict['body_text']:
            section_name = paragraph_dict['section']
            paragraph_text = paragraph_dict['text']
            # Check if we are still on the same section, or a new one.
            if section_name == last_section:
                paper_body_text += paragraph_text + '\n\n'
            # Check that the new section name is not empty.
            elif section_name:
                paper_body_text += '<< ' + section_name + ' >>\n' + paragraph_text + '\n\n'
            # Save the section name for the next iteration.
            last_section = section_name

    # Remove white spaces at the beginning and end file.
    paper_body_text = paper_body_text.strip()
    # The Body Text found on the JSON Paper.
    return paper_body_text


def select_body_text(body_text_list: list):
    """
    Given a list of the body texts of a Paper, select the best body text to
    represent the Paper.

    - Current Strategy: Select the text with the biggest size.

    Args:
        body_text_list: A list of the text extracted from the Body Text files
            of the paper.

    Returns:
        The best Text representation of the Body Texts.
    """
    # Check we have at least one body text.
    if not body_text_list:
        # No Text Found.
        return ''

    # Get the text with the biggest size.
    max_text = body_text_list[0]
    max_size = len(max_text)
    other_texts = body_text_list[1:]
    for new_text in other_texts:
        # Continue if this text is smaller.
        if len(new_text) <= max_size:
            continue
        # We have a new bigger text.
        max_text = new_text
        max_size = len(new_text)

    # The Biggest text in the list.
    return max_text


if __name__ == '__main__':
    # Record the Runtime of the Program
    stopwatch = TimeKeeper()

    # <<< Testing the Papers class >>>
    # Load the CORD-19 Dataset
    print("\nLoading the CORD-19 Dataset...")
    the_papers = PapersCord19(show_progress=True)
    print("Done.")
    print(f"[{stopwatch.formatted_runtime()}]")

    # Get the amount of documents the dataset has.
    num_papers = len(the_papers)
    print(f"\nThe current CORD-19 dataset has {big_number(num_papers)} documents.")

    # Get the 'cord_uid' of one of the papers.
    cord19_ids = the_papers.papers_cord_uids()
    rand_cord_uid = choice(cord19_ids)

    # # Getting the embedding of one of the papers.
    # print(f"\nGetting the Embedding for the Paper <{rand_cord_uid}>...")
    # result = the_papers.paper_embedding(rand_cord_uid)
    # print(f"The Embedding is:")
    # print(result)

    # Getting the text of one of the papers.
    print(f"\nGetting the content of the Paper <{rand_cord_uid}>...")
    result = the_papers.formatted_paper_content(rand_cord_uid)
    # Trim the size of the paper's content.
    if len(result) > 1_500:
        result = result[:1_500] + '...'
    print("The Content of the paper.")
    print(result)

    # Display the Information of the Paper.
    print(f"\nInformation of the Paper <{rand_cord_uid}>:")
    the_paper_info = the_papers.papers_index[rand_cord_uid]
    the_title = the_paper_info['title']
    the_time = the_paper_info['publish_time']
    the_authors = the_paper_info['authors']
    print(f"  Title: {the_title}")
    print(f"  Publish Time: {the_time}")
    print(f"  Authors: {the_authors}")

    # Test Formatted Paper's Author & Publish Date.
    print("\nFormatted Author Info: ")
    pprint(the_papers.paper_authors(rand_cord_uid))
    print("\nFormatted Publication Date:")
    pprint(the_papers.paper_publish_date(rand_cord_uid))

    # # Quit Loop (If Using a Loop to Test)..
    # user_input = input("\nType [q/quit] to Exit Loop.\n")
    # if user_input.lower().strip() in {'q', 'quit', 'exit'}:
    #     break

    # ---- Testing All Dates & Authors Available ----
    # # Check the format of the Dates & Authors of the Papers.
    # all_dates = [the_papers.paper_publish_date(the_doc_id) for the_doc_id in the_papers.papers_index]
    # print(f"\n {len(all_dates)} dates were processed. All Good.")
    # # Check the format of the Authors of the papers
    # all_authors = [the_papers.paper_authors(the_doc_id) for the_doc_id in the_papers.papers_index]
    # print(f"\n {len(all_authors)} authors were processed. All Good.")

    print("\nDone.")
    print(f"[{stopwatch.formatted_runtime()}]")
