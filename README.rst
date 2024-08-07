.. image:: https://github.com/tjsanti/sigalike/blob/main/assets/sigalike.jpeg?raw=true
   :width: 40%

|Python compat| |PyPi| |GHA tests| |Codecov report| |readthedocs|

.. inclusion-marker-do-not-remove

sigalike
==============

sigalike is a Python module that provides a simple and efficient way to calculate the shifted sigmoid similarity score between two strings. The shifted sigmoid similarity score acts as a fuzzy string matching metric, allowing for the comparison of strings with varying levels of similarity.

The module provides two main functions: `shifted_sigmoid_similarity` and `best_match`. The `shifted_sigmoid_similarity` function calculates the shifted sigmoid similarity score between two input strings, while the `best_match` function returns the best match(es) between two collections or a string and a collection based on the shifted sigmoid similarity score.

The module includes basic built-in preprocessing for the input strings, which removes punctuation and converts all characters to lowercase. This preprocessing step helps to improve the accuracy of the similarity score.

Overall, sigalike is a lightweight and easy-to-use tool for fuzzy string matching in Python. It can be useful in various applications, such as text classification, search engines, and data cleaning.


Installation
============

sigalike requires Python ``>=3.9`` and can be installed via:

.. code-block:: bash

   pip install sigalike


Usage
=====

.. code-block:: python

   from sigalike import shifted_sigmoid_similarity, best_match

   shifted_sigmoid_similarity("make up", "make up make up")  # 1.0
   shifted_sigmoid_similarity("lazy dog", "the quick brown fox jumps over the lazy dog", shift=8)  # 0.8807970779778824

   best_match("hello world", ["hello world", "goodbye world"])  # BestMatch(match='hello world', score=1.0)
   best_match(
       "hello world", ["goodbye world", "hi mom", "tell the world I said hello"], shift=2
   )  # BestMatch(match='goodbye world', score=0.2310585786300049)



.. |GHA tests| image:: https://github.com/tjsanti/sigalike/workflows/tests/badge.svg
   :target: https://github.com/tjsanti/sigalike/actions?query=workflow%3Atests
   :alt: GHA Status
.. |Codecov report| image:: https://codecov.io/github/tjsanti/sigalike/coverage.svg?branch=main
   :target: https://codecov.io/github/tjsanti/sigalike?branch=main
   :alt: Coverage
.. |readthedocs| image:: https://readthedocs.org/projects/sigalike/badge/?version=latest
        :target: https://sigalike.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status
.. |Python compat| image:: https://img.shields.io/badge/>=python-3.9-blue.svg
.. |PyPi| image:: https://img.shields.io/pypi/v/sigalike.svg
        :target: https://pypi.python.org/pypi/sigalike
