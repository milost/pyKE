# -*- coding: utf-8 -*-
import hashlib

import numpy as np
import pandas as pd


def split_nt_line(line: str):
    """
    Splits a line from a N-triples file into subject, predicate and object.

    :param line: Line from a N-triples file
    :return: tuple with subject, predicate, object
    """
    s, p, o = line.split(maxsplit=2)
    if s.startswith("<") and s.endswith('>'):
        s = s.lstrip("<").rstrip(">")
    if p.startswith("<") and p.endswith('>'):
        p = p.lstrip("<").rstrip(">")

    o = o.strip()
    if o.endswith(" ."):
        o = o.rstrip(" .")

    if o.startswith("<") and o.endswith('>'):
        o = o.lstrip("<").rstrip(">")
    return s, p, o


def md5(filename: str):
    """
    Returns the MD5-hashsum of a file.

    :param filename: Filename
    :return: MD5-hashsum of the file
    """
    hash_md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_array_pointer(a):
    """
    Returns the address of the numpy array.

    :param a: Numpy array
    :return: Memory address of the array
    """
    return a.__array_interface__['data'][0]


def get_rank(predictions: np.array, value: float):
    """
    Helper function. Returns the index of value in predictions, if predictions were sorted ascending.

    :param predictions: list of prediction values
    :param value: value to look for
    :return: index of the value (i.e. number of predictions smaller than value)
    """
    smaller_predictions = np.where(predictions < value)
    return len(smaller_predictions[0]) + 1


def calc_metrics(rank_predictions=None, k=10):
    """
    Computes mean rank and hits@k score
    :param rank_predictions:
    :param k:
    :return:
    """
    if isinstance(rank_predictions, str):
        rankings = pd.read_csv(rank_predictions)
    else:
        rankings = rank_predictions

    results = []
    column_headers = []
    column_names = [column_name for column_name in list(rankings) if column_name.endswith("_rank")]

    for column_name in column_names:
        column_headers.append(f'{column_name.rstrip("_rank")}_hits_at_{k}')
        column_headers.append(f'{column_name.rstrip("_rank")}_mean_hits_at_{k}')

        hits_at_n = len(rankings[column_name][rankings[column_name] < k])
        mean_hits_at_n = hits_at_n / len(rankings[column_name])
        results.append(hits_at_n)
        results.append(mean_hits_at_n)

    column_headers.append('mean_rank')
    rank_sum = rankings[column_names].sum().sum()
    results.append(rank_sum / (2 * len(rankings[column_names[0]])))

    return pd.DataFrame([results], columns=column_headers)
