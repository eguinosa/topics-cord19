# Gelin Eguinosa Rosique

import logging

from papers import Papers
from doc2vec_cord19 import Doc2VecCord19
from time_keeper import TimeKeeper
from extra_funcs import big_number


if __name__ == '__main__':
    # Start Logging, to show Doc2Vec progress.
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # Record the Runtime of the Program
    stopwatch = TimeKeeper()

    # Load Corpus.
    print(f"\nLoading CORD-19 corpus...")
    corpus_cord19 = Papers(show_progress=True)
    print("Done.")
    print(f"[{stopwatch.formatted_runtime()}]")

    # Check the amount of documents loaded.
    num_papers = len(corpus_cord19.papers_cord_uids())
    print(f"\nThe current CORD-19 dataset has {big_number(num_papers)} documents.")

    # ********************************************************************
    # <--- Create Doc2Vec model using all content of  CORD-19 corpus --->
    # (Done - Papers' All Content - 9 hours for 138,967 documents)
    # New Doc2Vec - [8 h : 7 min : 25 sec : 62 mill]
    # ********************************************************************

    # # Create Doc2Vec model.
    # print("\nCreating Doc2Vec model using all content...")
    # doc_model = Doc2VecCord19(corpus=corpus_cord19, vector_dims=300,
    #                           use_title_abstract=False, show_progress=True)
    # print("Done.")
    # print(f"[{stopwatch.formatted_runtime()}]")
    #
    # # Save CORD-19 Corpus Word2Vec Model.
    # print("\nSaving the Doc2Vec Model of the CORD-19 Dataset...")
    # doc_model.save_model('cord19_dataset')
    # print("Done.")
    # print(f"[{stopwatch.formatted_runtime()}]")

    # # ********************************************************************
    # # <--- Create Doc2Vec model using Titles & Abstracts of CORD-19 --->
    # # (Done - Papers' Title & Abstract - 35 min for 138,967 documents)
    # # ********************************************************************
    #
    # # Create Doc2Vec model.
    # print("\nCreating Doc2Vec model using Titles & Abstracts...")
    # doc_model = Doc2VecCord19(corpus=corpus_cord19, vector_dims=300,
    #                           use_title_abstract=True, show_progress=True)
    # print("Done.")
    # print(f"[{stopwatch.formatted_runtime()}]")
    #
    # # Save CORD-19 Corpus Word2Vec Model.
    # print("\nSaving the Doc2Vec Model of the CORD-19 Dataset...")
    # doc_model.save_model('cord19_title_abstract')
    # print("Done.")
    # print(f"[{stopwatch.formatted_runtime()}]")
