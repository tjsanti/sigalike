import numpy as np


def shifted_sigmoid_similarity(str1: str, str2: str) -> float:
    """Calculates the shifted sigmoid similarity score between two strings.

    The shifted sigmoid similarity score acts as a fuzzy string matching
    metric. It begins with the ratio of unique tokens from the shorter
    string that are present in the longer string and then subtracts a penalty
    using a shifted sigmoid function (the logistic function) and the difference
    in length between the two token sets.

    Parameters
    ----------
    str1 : str
        The first string to compare.
    str2 : str
        The second string to compare.

    Returns
    -------
    float
        Shifted sigmoid similarity score.
    """
    set_str1 = set(str1.lower().split())
    set_str2 = set(str2.lower().split())
    smaller_set = min(len(set_str1), len(set_str2))
    larger_set = max(len(set_str1), len(set_str2))

    num_matched = len(set_str1.intersection(set_str2))
    match_ratio = num_matched / smaller_set
    num_extra = larger_set - num_matched

    score = match_ratio
    # Penalize if non-perfect match
    if score > 0 and num_extra > 0:
        score -= 1 / (1 + np.exp(-num_extra + 4))

    return max(score, 0)
