# Gelin Eguinosa Rosique
# 2022

import json
import numpy as np
from os import mkdir
from os.path import isdir, isfile, join
from shutil import rmtree

from corpus_cord19 import CorpusCord19
from document_model import DocumentModel
from doc_tokenizers import doc_tokenizer
from extra_funcs import progress_bar


class Vocabulary:
    """
    Manage the vocabulary of the corpus creating an id for each word, saving
    the word embeddings and the documents' vocabulary, so we can see to which
    documents each word belongs to. It also has tools to calculate the
    frequencies of the words in the documents and the corpus.
    """
    # Class variables filenames.
    vocabulary_index = 'vocabulary_index.json'
    word2embed_file = 'word2embed.json'
    word2count_file = 'word2count.json'
    doc2content_file = 'word2content.json'

    def __init__(self, corpus: CorpusCord19 = None, doc_model: DocumentModel = None,
                 _load_vocab=False, _vocab_dir_path: str = None,
                 title_abstract=True, show_progress=False):
        """
        Create a dictionary with all the word in the corpus and their embeddings.
        Also, save the vocabulary for each document.

        Args:
            corpus: A Cord-19 Corpus class with the selection of papers in the
                corpus.
            doc_model: A Document Model class used to get the embeddings of the
                words in the corpus.
            _load_vocab: Bool indicating if we have to load the Corpus Vocab
                instead of creating it.
            _vocab_dir_path: String with the directory where the Corpus Vocab is
                stored.
            title_abstract: Bool indicating if we will only use the title and
                abstract, or the full content of the paper.
            show_progress: A Bool representing whether we show the progress of
                the function or not.
        """
        # Check if we have to load a saved version of the vocabulary.
        if _load_vocab:
            # Check if we have a valid '_vocab_dir_path':
            if not _vocab_dir_path or type(_vocab_dir_path) != 'str':
                raise Exception("No proper Path was provided to load the Vocabulary!!")
            if not isdir(_vocab_dir_path):
                raise Exception("There is not Vocabulary Folder at the provided Path!!")

            # Load class Index.
            index_path = join(_vocab_dir_path, self.vocabulary_index)
            if not isfile(index_path):
                raise Exception("The CorpusVocab Index file does not exist!!")
            with open(index_path, 'r') as f:
                vocab_index = json.load(f)
            corpus_length = vocab_index['corpus_length']
            vocab_size = vocab_index['vocab_size']
            # Progress Variables.
            count = 0
            total = vocab_size + 4
            if show_progress:
                count += 1
                progress_bar(count, total)

            # Load Word Count Dictionary.
            word2count_path = join(_vocab_dir_path, self.word2count_file)
            if not isfile(word2count_path):
                raise Exception("The CorpusVocab Word2Count file does not exist!!")
            with open(word2count_path, 'r') as f:
                word2count = json.load(f)
            if show_progress:
                count += 1
                progress_bar(count, total)

            # Load Documents' Bag of Words.
            doc2content_path = join(_vocab_dir_path, self.doc2content_file)
            if not isfile(doc2content_path):
                raise Exception("The CorpusVocab Doc's Bags-of-Words file does not exist!!")
            with open(doc2content_path, 'r') as f:
                doc2content = json.load(f)
            if show_progress:
                count += 1
                progress_bar(count, total)

            # Load Word Embeddings.
            word2embeds_path = join(_vocab_dir_path, self.word2embed_file)
            if not isfile(word2embeds_path):
                raise Exception("The CorpusVocab Word Embeddings file does not exist!!")
            with open(word2embeds_path, 'r') as f:
                word2embeds_index = json.load(f)
            # Transform embeddings back to Numpy.ndarray.
            word2embed = {}
            for word, embed in word2embeds_index:
                word2embed[word] = np.array(embed)
                if show_progress:
                    count += 1
                    progress_bar(count, total)
            # Last Progress Report.
            if show_progress:
                count += 1
                progress_bar(count, total)
        else:
            # Variables to save Corpus Vocabulary Info.
            corpus_length = 0
            word2count = {}
            doc2content = {}
            word2embed = {}

            # Progress Bar Variables.
            count = 0
            total = len(corpus)

            # Iterate through the papers and extract their content.
            for cord_uid in corpus.papers_cord_uids():
                # Check if we are only using the Title & Abstract of the papers.
                if title_abstract:
                    doc_content = corpus.paper_title_abstract(cord_uid)
                else:
                    doc_content = corpus.paper_content(cord_uid)
                # Tokenize the content of the paper.
                doc_tokens = doc_tokenizer(doc_content)

                # Create embeddings for the new words.
                new_words = [token for token in doc_tokens if token not in word2embed]
                new_embeds = doc_model.words_vectors(new_words)
                for new_word, new_embed in zip(new_words, new_embeds):
                    word2embed[new_word] = new_embed

                # Create Doc Bag-of-Words.
                doc_word2count = {}
                # Update Word count Info.
                for token in doc_tokens:
                    # Update the total count of tokens in the corpus.
                    corpus_length += 1
                    # Update the token count in the corpus.
                    if token not in word2count:
                        word2count[token] = 0
                    word2count[token] += 1
                    # Update the token count in the document.
                    if token not in doc_word2count:
                        doc_word2count[token] = 0
                    doc_word2count += 1

                # Save Doc Bag-of-Words.
                doc2content[cord_uid] = doc_word2count
                # Progress.
                if show_progress:
                    count += 1
                    progress_bar(count, total)

        # Save Vocabulary Info.
        self.corpus_length = corpus_length
        self.word2count = word2count
        self.doc2content = doc2content
        self.word2embed = word2embed

    def save(self, dir_path: str, show_progress=False):
        """
        Save the vocabulary's info.

        Args:
            dir_path: String with the path to the directory where the Vocabulary
                will be saved.
            show_progress: A Bool representing whether we show the progress of
                the function or not.
        """
        # Check if we have a valid 'dir_path'.
        if not dir_path or type(dir_path) != 'str':
            raise Exception("No proper Path to store the Vocabulary was provided!!")
        # Clean the directory where the vocabulary will be stored.
        if isdir(dir_path):
            rmtree(dir_path)
        mkdir(dir_path)
        # Progress Variables. (Add the 4 saving actions)
        count = 0
        total = len(self.word2embed) + 4

        # Save class index.
        vocab_index = {
            'corpus_length': self.corpus_length,
            'vocab_size': len(self.word2count)
        }
        index_path = join(dir_path, self.vocabulary_index)
        with open(index_path, 'w') as f:
            json.dump(vocab_index, f)
        if show_progress:
            count += 1
            progress_bar(count, total)

        # Save class word count dictionary.
        word2count_path = join(dir_path, self.word2count_file)
        with open(word2count_path, 'w') as f:
            json.dump(self.word2count, f)
        if show_progress:
            count += 1
            progress_bar(count, total)

        # Save Documents' Bag of words.
        doc2content_path = join(dir_path, self.doc2content_file)
        with open(doc2content_path, 'w') as f:
            json.dump(self.doc2content, f)
        if show_progress:
            count += 1
            progress_bar(count, total)

        # Save Word Embeddings transforming first the embeddings to list.
        word2embeds_index = {}
        for word, embed in self.word2embed.items():
            word2embeds_index[word] = embed.tolist()
            if show_progress:
                count += 1
                progress_bar(count, total)
        # Create Word Embeds Path & Save.
        word2embeds_path = join(dir_path, self.word2embed_file)
        with open(word2embeds_path, 'w') as f:
            json.dump(word2embeds_index, f)
        if show_progress:
            count += 1
            progress_bar(count, total)
