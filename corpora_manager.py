# Gelin Eguinosa Rosique

import json
import random
from os import mkdir, listdir
from shutil import rmtree
from os.path import join, isdir, isfile

from papers_cord19 import PapersCord19
from time_keeper import TimeKeeper
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
    default_sample_id = 'default'
    sample_prefix = 'random_sample_'

    def __init__(self, corpus_id='', corpus: PapersCord19 = None, sample_id='',
                 new_sample_size=-1, show_progress=False):
        """
        Create a random subset of the documents inside the provided 'corpus' and
        the requested 'size'. If '_load_corpus' is True, then load the previously
        saved corpus with the given '_corpus_id'.

        - Checks if we have a corpus saved. If not, saves the given 'corpus'
          or the default '2020-05-31'.
        - If a 'sample_id' is provided, loads this saved random sample. Otherwise,
          creates a new random sample with the value of the 'new_sample_size'.
        - If the 'new_sample_size' is -1, loads all the documents inside the
          corpus.

        Args:
            corpus_id: String with the ID of the corpus we are going to load.
            corpus: CorpusCord19 class containing all the papers we can use to
                create our random subset.
            sample_id: String with the ID of the Random Sample we want to
                load.
            new_sample_size: Int with the number of documents we want in our
                random subset.
            show_progress: Bool representing whether we show the progress of
                the function or not.
        """
        # Check if we have corpus ID.
        if not corpus_id:
            corpus_id = self.default_cord19_dataset
        # Create Paths to class attributes.
        corpus_folder_name = self.corpus_folder_prefix + corpus_id
        corpus_folder_path = join(self.corpora_data_folder, corpus_folder_name)
        corpus_index_path = join(corpus_folder_path, self.corpus_index_file)

        # Check if the corpus is saved.
        if self.corpus_saved(corpus_id=corpus_id):
            if show_progress:
                progress_msg(f"Corpus <{corpus_id}> available.")
            # Load Corpus Index.
            with open(corpus_index_path, 'r') as f:
                corpus_index = json.load(f)
        # Save the Corpus before doing anything else
        else:
            if show_progress:
                progress_msg(f"Corpus <{corpus_id}> not saved.")
            # Check we have a corpus to save.
            if not corpus:
                if show_progress:
                    progress_msg(f"Loading Papers from Corpus <{corpus_id}...")
                corpus = PapersCord19(dataset_id=corpus_id, show_progress=show_progress)
            # Create Corpora folder if it doesn't exist.
            if not isdir(self.corpora_data_folder):
                mkdir(self.corpora_data_folder)
            # Save corpus.
            if show_progress:
                progress_msg(f"Saving Corpus <{corpus_id}>...")
            corpus_index = self.save_corpus_data(folder_path=corpus_folder_path,
                                                 corpus=corpus,
                                                 show_progress=show_progress)
        # Done Loading the Index of the Corpus.
        if show_progress:
            progress_msg("Corpus Index Loaded.")

        # Get the IDs of the Papers we are going to use. (Random Sample)
        if sample_id:
            # Check if this Random Sample is saved.
            sample_file_name = self.sample_prefix + sample_id + '.json'
            sample_file_path = join(corpus_folder_path, sample_file_name)
            if not isfile(sample_file_path):
                raise Exception(f"There is not Random Sample saved with the name <{sample_id}>")
            # Load IDs of the Random Sample.
            with open(sample_file_path, 'r') as f:
                current_doc_ids = json.load(f)
        elif new_sample_size != -1 & new_sample_size < len(corpus_index):
            # Create a new Random Sample.
            all_doc_ids = list(corpus_index)
            current_doc_ids = random.sample(all_doc_ids, new_sample_size)
        else:
            # Using all the Documents in the Corpus.
            current_doc_ids = list(corpus_index)

        # Save class attributes.
        self.corpus_id = corpus_id
        self.doc_ids = current_doc_ids
        self.corpus_index = corpus_index
        # In case we need to load All Specter Embeddings.
        self.doc_embeds = None

    def __len__(self):
        """
        Length of the current corpus, depending on the size of the Random
        Sample requested.

        Returns: Int with the number of Documents in the Random Sample.
        """
        result = len(self.doc_ids)
        return result

    def doc_content(self, doc_id: str, full_content=False):
        """
        Get the content of the document. Title & Abstract by default, all the
        available content of the document if 'full_content' is True.

        Args:
            doc_id: String with the ID of the document.
            full_content: Bool indicating if we also need to include the body
                text in the content of the document.

        Returns:
            String with the content of the document.
        """
        # Get Document Dictionary.
        doc_info = self.corpus_index[doc_id]
        # Get Content.
        doc_title = doc_info['title']
        doc_abstract = doc_info['abstract']
        doc_content = doc_title + '\n' + doc_abstract
        if full_content and doc_info['body_text_path']:
            body_text_path = doc_info['body_text_path']
            with open(body_text_path, 'r') as f:
                body_text = f.read()
            doc_content += '\n' + body_text
        # The Content of the Document.
        return doc_content

    def doc_embed(self, doc_id: str):
        """
        Get the Specter Embedding of the Document.

        Args:
            doc_id: String with the ID of the document.

        Returns: List[float] with the Specter embedding of the Document.
        """
        # Check if the embeds Dict was loaded.
        if self.doc_embeds:
            embedding = self.doc_embeds[doc_id]
        else:
            # Upload the Embed from file.
            doc_info = self.corpus_index[doc_id]
            embedding_path = doc_info['specter_embed_path']
            with open(embedding_path, 'r') as f:
                embedding = json.load(f)

        # Specter Embedding.
        return embedding

    def load_embeddings_dict(self):
        """
        Load to memory the embeddings Dictionary (around 2GB). Speeds up the
        process to get the embedding of the documents.
        """
        # Path to Folder & Dictionary.
        corpus_folder_name = self.corpus_folder_prefix + self.corpus_id
        corpus_folder_path = join(self.corpora_data_folder, corpus_folder_name)
        corpus_embeds_path = join(corpus_folder_path, self.specter_embeds_file)
        # Load dictionary.
        with open(corpus_embeds_path, 'r') as f:
            self.doc_embeds = json.load(f)

    def unload_embeddings_dict(self):
        """
        Unload from memory the embeddings Dictionary. Frees up space.
        """
        self.doc_embeds = None

    def save_corpus_data(self, folder_path: str, corpus: PapersCord19,
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

            # Get Authors and Publication date.
            doc_authors = corpus.paper_authors(doc_id)
            doc_time = corpus.paper_publish_date(doc_id)

            # Get Length of the Doc.
            doc_length = len(doc_title_abstract + doc_body_text)
            # Create Document Index.
            document_index = {
                'cord_uid': doc_id,
                'title': doc_title,
                'abstract': doc_abstract,
                'authors': doc_authors,
                'publish_date': doc_time,
                'char_length': doc_length,
                'specter_embed_path': individual_embeds_file_path,
                'title_abstract_path': title_abstract_file_path,
                'body_text_path': body_text_file_path,
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
    def corpus_saved(cls, corpus_id=''):
        """
        Check if the given corpus is already saved. Checks for the default
        corpus if no 'corpus_id' is provided.

        Args:
            corpus_id: String with the ID of the corpus.

        Returns:
            Bool indicating if the corpus is available.
        """
        # Check we have a 'corpus_id'.
        if not corpus_id:
            corpus_id = cls.default_cord19_dataset

        # Check the class folder exists.
        if not isdir(cls.corpora_data_folder):
            return False
        # Check the corpus folder exists.
        corpus_folder_name = cls.corpus_folder_prefix + corpus_id
        corpus_folder_path = join(cls.corpora_data_folder, corpus_folder_name)
        if not isdir(corpus_folder_path):
            return False
        # Check for the Corpus Index File.
        corpus_index_path = join(corpus_folder_path, cls.corpus_index_file)
        if not isfile(corpus_index_path):
            return False

        # All good.
        return True

    @classmethod
    def available_corpora(cls):
        """
        Check inside the 'corpus_data' class folder and create a list with the
        ids of the corpora available.

        Returns: List[str] with the ids of the CorpusManager() that can be
            loaded.
        """
        # Check the corpora folder exists.
        if not isdir(cls.corpora_data_folder):
            return []

        # Create Default List.
        corpora_ids = []
        for filename in listdir(cls.corpora_data_folder):
            if cls.corpus_saved(corpus_id=filename):
                corpora_ids.append(filename)

        # List of Saved Corpora.
        return corpora_ids

    @classmethod
    def sample_saved(cls, corpus_id='', sample_id=''):
        """
        Check if a given Random Sample ID was already saved.

        Args:
            corpus_id: String with the ID of the corpus.
            sample_id: String with the ID of the Sample we want to check.

        Returns:
            Bool indicating if the Random sample is available or not.
        """
        # Check if we have a Corpus ID.
        if not corpus_id:
            corpus_id = cls.default_cord19_dataset
        # Check the Sample ID.
        if not sample_id:
            sample_id = cls.default_sample_id

        # Check the corpora.
        if not isdir(cls.corpora_data_folder):
            return False
        # Check corpus folder.
        corpus_folder_name = cls.corpus_folder_prefix + corpus_id
        corpus_folder_path = join(cls.corpora_data_folder, corpus_folder_name)
        if not isdir(corpus_folder_path):
            return False
        # Check the Random Sample file.
        sample_file_name = cls.sample_prefix + sample_id + '.json'
        sample_file_path = join(corpus_folder_path, sample_file_name)
        if not isfile(sample_file_path):
            return False

        # All Good.
        return True

    @classmethod
    def available_samples(cls, corpus_id=''):
        """
        Check the saved Random Samples for the given 'corpus_id'.

        Args:
            corpus_id: String with the ID of the corpus.

        Returns:
            List[str] with Random Sample's IDs that are available.
        """
        # Check we have a Corpus ID.
        if not corpus_id:
            corpus_id = cls.default_cord19_dataset
        # Check we have a saved Corpus ID.
        if not cls.corpus_saved(corpus_id=corpus_id):
            return []

        # Create path to the folder of the corpus ID.
        corpus_folder_name = cls.corpus_folder_prefix + corpus_id
        corpus_folder_path = join(cls.corpora_data_folder, corpus_folder_name)
        # Search for the saved Random Samples.
        saved_samples = []
        for filename in listdir(corpus_folder_path):
            if not isfile(filename):
                continue
            if not filename.startswith(cls.sample_prefix):
                continue
            if not filename.endswith('.json'):
                continue
            # We have new sample.
            sample_id = filename.replace('.json', '')
            saved_samples.append(sample_id)

        # The Samples found.
        return saved_samples


if __name__ == '__main__':
    # Record Runtime of the Program.
    stopwatch = TimeKeeper()

    print("\nDone.")
    print(f"[{stopwatch.formatted_runtime()}]\n")
