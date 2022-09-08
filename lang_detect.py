# Gelin Eguinosa Rosique
# 2022

import fasttext
from os.path import isdir, isfile, join

# Test Imports.
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
        language.

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


if __name__ == '__main__':
    # Keep track of runtime.
    stopwatch = TimeKeeper()

    # Create instance of class.
    the_detector = LangDetect()

    # Predict the Language of Texts.
    while True:
        the_input = input("\nType a text to predict Language (q/quit to exit).\n-> ")
        if the_input.lower().strip() in {'', 'q', 'quit', 'exit'}:
            break
        # the_prediction = the_detector.detect_languages(the_input, k=1)
        the_prediction = the_detector.text_language(the_input)
        print("\nThe language of the text is:")
        print(the_prediction)

    print("\nDone.")
    print(f"[{stopwatch.formatted_runtime()}]\n")
