# Gelin Eguinosa Rosique
# 2022

import json
from os import mkdir, listdir
from shutil import rmtree
from os.path import join, isdir, isfile

from corpus_cord19 import CorpusCord19
from papers_cord19 import PapersCord19
from lang_detect import LangDetect
from extra_funcs import progress_bar, progress_msg, big_number

# Testing Imports.
# from pprint import pprint
from random_sample import RandomSample
from time_keeper import TimeKeeper


class CorporaManager:
    """
    Class to manage the documents, document's embeddings and vocabulary's
    embeddings belonging to a Corpus.
    """
    # Class Data Names & Locations.
    corpora_data_folder = 'corpora_data'
    default_corpus_id = 'default_corpus'
    default_cord19_dataset = '2020-05-31'
    corpus_folder_prefix = 'corpus_cord19_'
    title_abstracts_folder = 'title_abstracts'
    body_texts_folder = 'body_texts'
    single_embeds_folder = 'single_specter_embeddings'
    specter_embeds_file = 'specter_embeddings.json'
    corpus_index_file = 'corpus_index.json'

    def __init__(self, corpus_id='', corpus: CorpusCord19 = None, show_progress=False):
        """
        Extract from a Corpus of the Cord19 Dataset the viable documents for
        Topic Modeling, saving all their metadata of interest.

        It only saves the documents with non-empty title & abstract and
        eliminates the documents with content not in English.

        If 'corpus_id' is not already saved, it uses 'corpus' to create a new
        filtered corpus. If 'corpus' is empty, then loads 'default_cord19_dataset'.

        Args:
            corpus_id: String with the ID of the corpus we are going to load.
            corpus: CorpusCord19 class containing all the papers we can use to
                create our random subset.
            show_progress: Bool representing whether we show the progress of
                the function or not.
        """
        # Check if we have corpus ID.
        if not corpus_id:
            corpus_id = self.default_corpus_id
        # Create Paths to class attributes.
        corpus_folder_name = self.corpus_folder_prefix + corpus_id
        corpus_folder_path = join(self.corpora_data_folder, corpus_folder_name)
        corpus_index_path = join(corpus_folder_path, self.corpus_index_file)

        # Check if the corpus is saved.
        if self.corpus_saved(corpus_id=corpus_id):
            if show_progress:
                progress_msg(f"Corpus <{corpus_id}> available...")
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
                dataset_id = self.default_cord19_dataset
                corpus = PapersCord19(dataset_id=dataset_id, show_progress=show_progress)
            # Create Corpora folder if it doesn't exist.
            if not isdir(self.corpora_data_folder):
                mkdir(self.corpora_data_folder)
            # Save corpus.
            if show_progress:
                progress_msg(f"Saving Corpus <{corpus_id}>...")
            corpus_index = self.save_corpus_data(folder_path=corpus_folder_path,
                                                 corpus=corpus,
                                                 show_progress=show_progress)
        # Save in a list the IDs of all available Documents in the Corpus.
        corpus_doc_ids = list(corpus_index)
        # Done Loading the Index of the Corpus.
        if show_progress:
            progress_msg("Corpus Index Loaded.")

        # Save corpus attributes.
        self.corpus_id = corpus_id
        self.corpus_index = corpus_index
        self.corpus_doc_ids = corpus_doc_ids
        # In case we need to load All Specter Embeddings.
        self.doc_embeds = None

    def __len__(self):
        """
        The amount of documents inside the currently loaded corpus.

        Returns: Int with the number of Documents in the current Corpus.
        """
        result = len(self.corpus_doc_ids)
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
        single_embeds_folder_path = join(folder_path, self.single_embeds_folder)
        mkdir(single_embeds_folder_path)

        # Progress Variables.
        count = 0
        total = len(corpus) + 2  # saving index & embeddings dict.
        without_title = 0
        without_abstract = 0
        with_text = 0
        not_english_doc = 0
        # Create Language Checker to save only Docs in English.
        lang_detector = LangDetect()
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

            # Get Body Text.
            doc_body_text = corpus.paper_body_text(doc_id)
            # Check the document is in English.
            in_english = lang_detector.doc_in_english(title=doc_title, abstract=doc_abstract,
                                                      body_text=doc_body_text)
            # Skip Documents that are not in English.
            if not in_english:
                not_english_doc += 1
                if show_progress:
                    count += 1
                    progress_bar(count, total)
                continue

            # Save Title & Abstract.
            doc_title_abstract = doc_title + '\n\n' + doc_abstract
            title_abstract_filename = doc_id + '.txt'
            title_abstract_file_path = join(title_abstract_folder_path, title_abstract_filename)
            with open(title_abstract_file_path, 'w') as f:
                print(doc_title_abstract, file=f)
            # Save Body Text. Check we have a Body Text before Saving.
            if doc_body_text:
                with_text += 1
                body_text_filename = doc_id + '.txt'
                body_text_file_path = join(body_text_folder_path, body_text_filename)
                with open(body_text_file_path, 'w') as f:
                    print(doc_body_text, file=f)
            else:
                # Without Body Text.
                body_text_file_path = ''

            # Get & Save Embedding.
            doc_embed = corpus.paper_embedding(doc_id)
            # Save embed in Dictionary.
            doc_embeddings[doc_id] = doc_embed
            # Save Embed in Individual File.
            single_embeds_filename = doc_id + '.json'
            single_embeds_file_path = join(single_embeds_folder_path, single_embeds_filename)
            with open(single_embeds_file_path, 'w') as f:
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
                'specter_embed_path': single_embeds_file_path,
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
            progress_msg(f"{big_number(not_english_doc)} docs not in english.")
            progress_msg(f"{big_number(with_text)} docs with body text.")
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
            corpus_id = cls.default_corpus_id

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
            # Check if this a viable corpus folder.
            prefix_len = len(cls.corpus_folder_prefix)
            if len(filename) <= prefix_len:
                continue
            # Remove the Corpus Folder Prefix from their name.
            corpus_id = filename[prefix_len:]
            if cls.corpus_saved(corpus_id=corpus_id):
                corpora_ids.append(corpus_id)

        # List of Saved Corpora.
        corpora_ids.sort()
        return corpora_ids


if __name__ == '__main__':
    # Record Runtime of the Program.
    stopwatch = TimeKeeper()

    # Load Random Sample with ID.
    # ----------------------------------------------------------
    # the_id = 'full_dataset'
    # print("\nLoading the Cord-19 Papers dataset...")
    # the_sample = PapersCord19(show_progress=True)
    # ----------------------------------------------------------
    the_id = '1000_docs'
    print(f"\nLoading Saved Random Sample <{the_id}>...")
    the_sample = RandomSample.load(sample_id=the_id, show_progress=True)
    # ----------------------------------------------------------
    print("Done.")
    print(f"[{stopwatch.formatted_runtime()}]")

    # Create a Corpora Manager with this Random Sample and the same ID.
    the_new_id = the_id
    the_size = len(the_sample)
    print(f"\nCreating Corpora Manager with {the_size} documents...")
    the_manager = CorporaManager(corpus_id=the_new_id, corpus=the_sample, show_progress=True)
    print("Done.")
    print(f"[{stopwatch.formatted_runtime()}]")

    # # Print Info of the First Document.
    # the_doc_id = the_manager.doc_ids[0]
    # the_doc_info = the_manager.corpus_index[the_doc_id]
    # print(f"\nInformation of the Document <{the_doc_id}>:")
    # pprint(the_doc_info)

    # Check the available Corpora.
    the_corpora_list = CorporaManager.available_corpora()
    if the_corpora_list:
        print("\nIDs of Available Corpora:")
    else:
        print("\nThere is no Corpora Saved and Ready to be Loaded.")
    for the_corpus_id in the_corpora_list:
        print(f"  -> {the_corpus_id}")

    print("\nDone.")
    print(f"[{stopwatch.formatted_runtime()}]\n")
