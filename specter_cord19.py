# Gelin Eguinosa Rosique

from transformers import AutoTokenizer, AutoModel

from document_model import DocumentModel
from time_keeper import TimeKeeper


class SpecterCord19(DocumentModel):
    """
    Load and Manage the Specter Model to get the vector representations of the
    word and documents in the CORD-19 corpus.
    """
    def __init__(self):
        """
        Load the tokenizer and model of specter.
        """
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
        self.model = AutoModel.from_pretrained('allenai/specter')

    def model_type(self):
        """
        Give the type of Document Model this class is: Specter

        Returns: A string with the model type.
        """
        return 'specter'

    def word_vector(self, word):
        """
        Give the embedding for the given 'word'.

        Args:
            word: A string with the word we want to encode.

        Returns:
            A Tensor containing the embedding of the word.
        """
        word_inputs = self.tokenizer([word], padding=True, truncation=True,
                                     return_tensors="pt", max_length=512)
        input_embeds = self.model(**word_inputs)
        word_embed = input_embeds.last_hidden_state[:, 0, :][0]
        return word_embed

    def document_vector(self, title_abstract):
        """
        Create the embedding of a paper providing the title and abstract of the
        document.

        Args:
            title_abstract: A string containing the title and abstract of the
                paper.

        Returns:
           A tensor with the embedding of the document.
        """
        doc_inputs = self.tokenizer([title_abstract], padding=True, truncation=True,
                                    return_tensors="pt", max_length=512)
        input_embeds = self.model(**doc_inputs)
        doc_embed = input_embeds.last_hidden_state[:, 0, :][0]
        return doc_embed


if __name__ == '__main__':
    # Record the Runtime of the Program
    stopwatch = TimeKeeper()

    print("\nCreating Specter Model...")
    specter_model = SpecterCord19()
    print("Done.")
    print(f"[{stopwatch.formatted_runtime()}]")

    # Get the embeddings from input words.
    while True:
        input_word = input("\nFrom which word do you want to get the embedding?\n ")
        if not input_word or input_word in {'q', 'quit'}:
            break
        print(f"Word vector of {input_word}:")
        print(specter_model.word_vector(input_word))

    # # load model and tokenizer
    # print("\nLoading or Downloading Specter Model...")
    # tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
    # model = AutoModel.from_pretrained('allenai/specter')
    # print("Done.")
    # print(f"[{stopwatch.formatted_runtime()}]")
    #
    # papers = [{'title': 'BERT', 'abstract': 'We introduce a new language representation model called BERT'},
    #           {'title': 'Attention is all you need',
    #            'abstract': 'The dominant sequence transduction models are based on complex recurrent or convolutional '
    #                        'neural networks'}]
    #
    # print("\nEncoding Documents...")
    # # concatenate title and abstract
    # title_abs = [d['title'] + tokenizer.sep_token + (d.get('abstract') or '') for d in papers]
    #
    # # preprocess the input
    # inputs = tokenizer(title_abs, padding=True, truncation=True, return_tensors="pt", max_length=512)
    # result = model(**inputs)
    # print("Done.")
    # print(f"[{stopwatch.formatted_runtime()}]")
    #
    # # take the first token in the batch as the embedding
    # embeddings = result.last_hidden_state[:, 0, :]
    #
    # for i, embed in enumerate(embeddings):
    #     print(f"\nThe embedding of the paper {i}:")
    #     print(embed)

