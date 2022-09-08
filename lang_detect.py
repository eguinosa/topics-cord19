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

    def detect_lang(self, text: str, langs=1):
        """
        Detect the language spoken on the 'text'.

        Args:
            text: String with the text we want to translate.
            langs: String with the languages you want to predict for the text.

        Returns:
            String with the most likely language spoken in the text.
        """
        prediction = self.model.predict(text, k=langs)
        return prediction


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
        the_prediction = the_detector.detect_lang(the_input, langs=2)
        print("\nThe language of the text is:")
        print(the_prediction)

    print("\nDone.")
    print(f"[{stopwatch.formatted_runtime()}]\n")
