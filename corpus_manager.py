# Gelin Eguinosa Rosique

import json
import random
from os import mkdir
from shutil import rmtree
from os.path import join, isdir, isfile

from corpus_cord19 import CorpusCord19
from papers_cord19 import PapersCord19
from extra_funcs import progress_bar, progress_msg, big_number


class CorporaManager:
    """
    Class to manage the documents, document's embeddings and vocabulary's
    embeddings belonging to a Corpus.
    """
    # Class Data Locations.
    corpora_data_folder = 'corpora_data'
    default_cord19_dataset = '2020-05-31'
    corpus_folder_prefix = 'corpus_cord19_'
    title_abstracts_folder = 'title_abstracts'
    body_texts_folder = 'body_texts'
    individual_embeds_folder = 'individual_specter_embeddings'
    specter_embeds_file = 'specter_embeddings.json'
    corpus_index_file = 'corpus_index.json'
    default_sample_file = 'random_sample_default.json'
    random_sample_prefix = 'random_sample_'

    def __init__(self, corpus_id='', corpus: CorpusCord19 = None, size=-1,
                 random_sample_id='', show_progress=False):
        """
        Create a random subset of the documents inside the provided 'corpus' and
        the requested 'size'. If '_load_corpus' is True, then load the previously
        saved corpus with the given '_corpus_id'.

        By default, creates a corpus the Cord19 Papers if no 'corpus' is given,
        with all the papers if 'size' is -1.

        Args:
            corpus_id: String with the ID of the corpus we are going to load.
            corpus: CorpusCord19 class containing all the papers we can use to
                create our random subset.
            size: Int with the number of documents we want in our random subset.
            random_sample_id: String with the ID of the Random Sample we want to
                load.
            show_progress: Bool representing whether we show the progress of
                the function or not.
        """
        # Check if we are loading or creating a corpus from scratch.
        if corpus_id in self.available_corpora():
            pass
        # Creating a new Corpus.
        else:
            # Report we are creating new corpus.
            if show_progress:
                progress_msg("Creating a new Corpus.")
            # Check if the source corpus was provided.
            if not corpus:
                if show_progress:
                    progress_msg("No source provided. Loading Cord19 dataset...")
                corpus = PapersCord19(show_progress=show_progress)

            # Get the cord_uids of the Documents we are going to add to our corpus.
            if size == -1:
                doc_ids = corpus.papers_cord_uids()
            else:
                corpus_size = min(size, len(corpus))
                doc_ids = random.sample(corpus.papers_cord_uids(), corpus_size)
            if show_progress:
                progress_msg(f"Loading {len(doc_ids)} documents...")

            # Extract the content of the papers & save them.
            # Here...

        # Save the IDs of the Documents.
        self.doc_ids = doc_ids

    def save_corpus_data(self, folder_path: str, corpus: CorpusCord19,
                         show_progress=False):
        """
        Save the title & abstracts, full content & embeddings of the papers in
        the provided 'corpus'.

        Args:
            folder_path: String with the path of the folder where all the data
                will be stored.
            corpus: CorpusCord19 with the document data we are going to store.
            show_progress: Bool representing whether we show the progress of
                the function or not.

        Returns:
            Dictionary with the index containing the location of all the data
                stored.
        """
        # Clean the folder if there was something there before.
        if isdir(folder_path):
            rmtree(folder_path)
        # Create a new folder for the corpus.
        mkdir(folder_path)

        # Create Folders for the documents' data.
        title_abstract_folder_path = join(folder_path, self.title_abstracts_folder)
        mkdir(title_abstract_folder_path)
        body_text_folder_path = join(folder_path, self.body_texts_folder)
        mkdir(body_text_folder_path)
        individual_embeds_folder_path = join(folder_path, self.individual_embeds_folder)
        mkdir(individual_embeds_folder_path)

        # Progress Variables.
        count = 0
        total = len(corpus) + 2  # saving index & embeddings dict.
        without_title = 0
        without_abstract = 0
        without_text = 0
        # Create Dictionary Index of the documents.
        docs_index = {}
        doc_embeddings = {}
        # Get & Save the documents' data.
        for doc_id in corpus.papers_cord_uids():
            # Extract Title & Abstract.
            doc_title = corpus.paper_title(doc_id)
            doc_abstract = corpus.paper_abstract(doc_id)
            # Check if we have both data fields.
            if not doc_title or not doc_abstract:
                if not doc_title:
                    without_title += 1
                if not doc_abstract:
                    without_abstract += 1
                if show_progress:
                    count += 1
                    progress_bar(count, total)
                # Skip when we don't have Title or Abstract. They are important.
                continue
            # Save Title & Abstract.
            doc_title_abstract = doc_title + '\n\n' + doc_abstract
            title_abstract_file_path = join(title_abstract_folder_path, doc_id)
            with open(title_abstract_file_path, 'w') as f:
                print(doc_title_abstract, file=f)

            # Get & Save Body Text.
            doc_body_text = corpus.paper_body_text(doc_id)
            # Check the Doc has Body Text.
            if doc_body_text:
                body_text_file_path = join(body_text_folder_path, doc_id)
                with open(body_text_file_path, 'w') as f:
                    print(doc_body_text, f)
            else:
                # Without Body Text.
                without_text += 1
                body_text_file_path = ''

            # Get & Save Embedding.
            doc_embed = corpus.paper_embedding(doc_id)
            # Save embed in Dictionary.
            doc_embeddings[doc_id] = doc_embed
            # Save Embed in Individual File.
            individual_embeds_file_path = join(individual_embeds_folder_path, doc_id)
            with open(individual_embeds_file_path, 'w') as f:
                json.dump(doc_embed, f)

            # Get Length of the Doc.
            doc_length = len(doc_title_abstract + doc_body_text)
            # Create Document Index.
            document_index = {
                'cord_uid': doc_id,
                'title_abstract': title_abstract_file_path,
                'body_text': body_text_file_path,
                'char_length': doc_length
            }
            # Save Document Index.
            docs_index[doc_id] = document_index

            # Progress.
            if show_progress:
                count += 1
                progress_bar(count, total)

        # Save Embeddings Dictionary.
        embeds_dict_path = join(folder_path, self.specter_embeds_file)
        with open(embeds_dict_path, 'w') as f:
            json.dump(doc_embeddings, f)
        if show_progress:
            count += 1
            progress_bar(count, total)

        # Save Doc's Dictionary.
        index_dict_path = join(folder_path, self.corpus_index_file)
        with open(index_dict_path, 'w') as f:
            json.dump(docs_index, f)
        if show_progress:
            count += 1
            progress_bar(count, total)
            # Final Progress Report.
            progress_msg("<------------------>")
            docs_saved = big_number(len(docs_index))
            total_docs = big_number(len(corpus))
            progress_msg(f"{docs_saved} documents out of {total_docs} saved.")
            progress_msg(f"{big_number(without_title)} docs without title.")
            progress_msg(f"{big_number(without_abstract)} docs without abstract.")
            progress_msg(f"{big_number(without_text)} docs without body text.")
            progress_msg("<------------------>")

        # Index with Documents' Info.
        return docs_index

    @classmethod
    def available_corpora(cls):
        """
        Check inside the 'corpus_data' class folder and create a list with the
        ids of the corpora available.

        Returns: List[str] with the ids of the CorpusManager() that can be
            loaded.

        """
        pass
