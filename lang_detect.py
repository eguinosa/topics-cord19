# Gelin Eguinosa Rosique
# 2022

import fasttext
from collections import defaultdict
from os.path import isdir, isfile, join

from corpus_cord19 import CorpusCord19
from extra_funcs import progress_bar

# Test Imports.
# from pprint import pprint
from random_sample import RandomSample
# from papers_cord19 import PapersCord19
from extra_funcs import big_number
from time_keeper import TimeKeeper


class LangDetect:
    """
    Class to detect the most likely language of a Text.
    """
    # Class locations.
    model_folder = 'fasttext_models'
    model_file = 'fasttext_model[lid.176].bin'

    # ISO of Languages of Interest (Fasttext supports a lot of languages).
    iso_dict = {
        'ar': 'arabic',
        'bn': 'bengali',
        'de': 'german',
        'en': 'english',
        'es': 'spanish',
        'fr': 'french',
        'hi': 'hindi',
        'it': 'italian',
        'ja': 'japanese',
        'ko': 'korean',
        'pt': 'portuguese',
        'pa': 'punjabi',
        'ru': 'russian',
        'zh': 'chinese',
    }

    def __init__(self):
        """
        Load fasttext model from a local file.
        """
        # Check the model folder exists.
        if not isdir(self.model_folder):
            raise FileNotFoundError("There is no fasttext folder to load the model from.")
        # Check the model file is available.
        model_path = join(self.model_folder, self.model_file)
        if not isfile(model_path):
            raise FileNotFoundError("There is not fasttext model available locally.")

        # Load and Save model.
        model = fasttext.load_model(model_path)
        self.model = model

    def text_language(self, text: str):
        """
        Detect the most likely language in 'text'.

        Args:
            text: String with the text we want to investigate.

        Returns:
            String with the most likely language of the text.
        """
        # Check the 'text' is not empty.
        if not text:
            raise ValueError("The text is empty.")

        # Get the Language (in a Tuple).
        languages, _ = self.model.predict(text)
        # Extract Language from the tuple.
        language = languages[0]
        # Delete the label at the beginning of the string.
        iso_lang = language[9:]
        # Get Full Name from the ISO if possible.
        full_lang = self.iso_dict.get(iso_lang, iso_lang)

        # Full Name of the Language.
        return full_lang

    def detect_languages(self, text: str, k=1):
        """
        Detect the languages spoken on the 'text' and the probability of these
        languages.

        Args:
            text: String with the text we want to translate.
            k: Int with the languages you want to predict for the text.

        Returns:
            Tuple(string, float) with the languages and their probabilities.
        """
        # Check the 'text' is not empty.
        if not text:
            raise ValueError("The text is empty.")

        # Predict the languages.
        langs, percents = self.model.predict(text, k=k)

        # Organize the result information.
        text_languages = []
        for lang, percent in zip(langs, percents):
            lang_iso = lang[9:]
            lang_name = self.iso_dict.get(lang_iso, lang_iso)
            text_languages.append((lang_name, percent))

        # Text Languages and their Probabilities
        return text_languages

    def doc_in_english(self, title, abstract, body_text):
        """
        Check if a Paper is in English. To qualify as an English Paper, it needs
        to have not empty title & abstract (body can be empty), and all the
        non-empty texts need to be in English.

        Args:
            title: String with the text of the title.
            abstract: String with the text of the abstract.
            body_text: String with the body text of the Paper.

        Returns:
            Bool indicating if the paper is in English or not.
        """
        # Check we have at least title & abstract.
        if not title or not abstract:
            return False

        # Check Title.
        title_lang = self.text_language(title)
        if title_lang != self.iso_dict['en']:
            return False
        # Check Abstract.
        abstract_lang = self.text_language(abstract)
        if abstract_lang != self.iso_dict['en']:
            return False
        # Check Body Text.
        if body_text:
            body_parags = body_text.split('\n')
            first_parag = body_parags[0]
            body_lang = self.text_language(first_parag)
            if body_lang != self.iso_dict['en']:
                return False

        # All Good. Everything is in English.
        return True

    def language_count(self, corpus: CorpusCord19, show_progress=False):
        """
        Count the Languages that appear on 'corpus'. Creates a dictionary with
        the languages of the corpus in the format:
            - (title_lang, abstract_lang) -> count
            - (title_lang, abstract_lang, body_text_lang) -> count
        The format depends on the information available about the document.

        Args:
            corpus: Corpus we are going to process.
            show_progress: Bool representing whether we show the progress of
                the function or not.

        Returns:
            Dict[Tuple -> Int] with the languages that appear on the corpus.
        """
        # Create Default Dictionary to store the counters.
        lang_counter = defaultdict(int)

        # Progress Variables.
        count = 0
        total = len(corpus)

        # Get the Languages of the Papers.
        for doc_id in corpus.papers_cord_uids():
            # Extract Content.
            doc_title = corpus.paper_title(doc_id)
            doc_abstract = corpus.paper_abstract(doc_id)
            doc_body = corpus.paper_body_text(doc_id)

            # Skip Documents with empty Title or Abstract.
            if doc_title and doc_abstract:
                # Get Doc Languages.
                title_lang = self.text_language(doc_title)
                abstract_lang = self.text_language(doc_abstract)

                # Check if the Document has Body Text.
                if doc_body:
                    # Get the first paragraph of the Body Text.
                    doc_paragraphs = doc_body.split('\n')
                    first_paragraph = doc_paragraphs[0]
                    body_lang = self.text_language(first_paragraph)
                    # Update Counter for documents with these attributes.
                    lang_counter[(title_lang, abstract_lang, body_lang)] += 1
                else:
                    # Update Counter with only title & abstract.
                    lang_counter[(title_lang, abstract_lang)] += 1

            if show_progress:
                count += 1
                progress_bar(count, total)

        # The Counter for the Languages in the Corpus.
        lang_counter = dict(lang_counter)
        return lang_counter


if __name__ == '__main__':
    # Keep track of runtime.
    stopwatch = TimeKeeper()

    # Create instance of class.
    the_detector = LangDetect()

    # Predict the Language of Texts.
    # while True:
    #     the_input = input("\nType a text to predict Language (q/quit to exit).\n-> ")
    #     if the_input.lower().strip() in {'', 'q', 'quit', 'exit'}:
    #         break
    #     # the_prediction = the_detector.detect_languages(the_input, k=1)
    #     the_prediction = the_detector.text_language(the_input)
    #     print("\nThe language of the text is:")
    #     print(the_prediction)

    # # ---- Check the Languages of the Texts in the Cord-19 Dataset ----
    # # ---------------------------------------------------------
    # Get the Corpus for the Test.
    sample_id = '3000_docs'
    print(f"\nLoading Saved Random Sample <{sample_id}>...")
    the_papers = RandomSample.load(sample_id=sample_id, show_progress=True)
    # # ---------------------------------------------------------
    # # Use CORD-19 Dataset
    # print("\nLoading the CORD-19 Dataset...")
    # the_papers = PapersCord19(show_progress=True)
    # # ---------------------------------------------------------
    # print("\nDetecting the Languages of the Papers...")
    # the_count = the_detector.language_count(the_papers, show_progress=True)
    # print("\nThe Languages of the Corpus:")
    # pprint(the_count)
    # # ---------------------------------------------------------
    # # Show how many papers are in english.
    # the_total = sum(the_count.values())
    # the_english_count = the_count[('english', 'english')] + the_count[('english', 'english', 'english')]
    # the_percent = the_english_count * 100 / the_total
    # the_percent = round(the_percent, 2)
    # print(f"\nPapers in English: {big_number(the_english_count)}")
    # print(f"Papers in other Languages: {big_number(the_total - the_english_count)}")
    # print(f"Percentage of Paper in english: {the_percent}")

    # ---- Test a Method to see if a Paper is in English ----
    # Create lists for the paper's ID depending on their languages.
    the_english_docs = []
    the_other_docs = []
    for the_doc_id in the_papers.papers_cord_uids():
        # Check if the paper is in English.
        the_title = the_papers.paper_title(the_doc_id)
        the_abstract = the_papers.paper_abstract(the_doc_id)
        the_body = the_papers.paper_body_text(the_doc_id)
        the_check = the_detector.doc_in_english(the_title, the_abstract, the_body)
        # Save Paper's ID on a list depending on its language.
        if the_check:
            the_english_docs.append(the_doc_id)
        else:
            the_other_docs.append(the_doc_id)
    # Report the amount of documents in English.
    print(f"\n{big_number(len(the_english_docs))} documents in English.")
    print(f"{big_number(len(the_other_docs))} documents in other Languages.")

    print("\nDone.")
    print(f"[{stopwatch.formatted_runtime()}]\n")
