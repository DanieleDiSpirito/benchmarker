"""Vendored copy of AllenAI's IFBench verifiers (Apache License 2.0, see LICENSE).

Source: https://github.com/allenai/IFBench (ifbench/ and evaluation_lib.py)
Changes: package-relative imports; NLTK data read from <project>/data/nltk; the
pickled punkt sentence tokenizer replaced by punkt_tab; no downloads at import time.
"""
