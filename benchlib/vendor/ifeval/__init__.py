"""Vendored copy of Google Research's IFEval checker (Apache License 2.0).

Source: https://github.com/google-research/google-research/tree/master/instruction_following_eval
Changes: package-relative imports; the sentence tokenizer uses NLTK's punkt_tab
instead of the pickled punkt model (refused by recent NLTK releases).
"""
