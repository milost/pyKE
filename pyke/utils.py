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


def get_rank_old(predictions: np.array, value: float):
    """
    Helper function. Returns the index of value in predictions, if predictions were sorted ascending.

    :param predictions: list of prediction values
    :param value: value to look for
    :return: index of the value (i.e. number of predictions smaller than value)
    """
    return (predictions > value).sum() + 1


def calc_hits_at_n(rankings:pd.DataFrame, k=1):

    head_k = (rankings.head_rank <= k).astype(int)
    tail_k = (rankings.tail_rank <= k).astype(int)
    head_hits_at_k = head_k.sum()
    tail_hits_at_k = tail_k.sum()

    total_hits_at_k = (head_k + tail_k).sum()
    factor = 100 / (2 * len(rankings))
    return head_hits_at_k, tail_hits_at_k, factor * total_hits_at_k


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
    column_headers = [
        'mean_head_rank',
        'mean_tail_rank',
        'mean_rank',
        'mean_reciprocal_head_rank',
        'mean_reciprocal_tail_rank',
        'mrr',
        f'head_hits_at_1',
        f'tail_hits_at_1',
        f'hits_at_1',
        f'head_hits_at_3',
        f'tail_hits_at_3',
        f'hits_at_3',
        f'head_hits_at_10',
        f'tail_hits_at_10',
        f'hits_at_10',
    ]

    mean_head_rank = rankings.head_rank.sum() / len(rankings)
    mean_tail_rank = rankings.tail_rank.sum() / len(rankings)
    mean_rank = (mean_head_rank + mean_tail_rank) / 2

    mean_head_r_rank = (1 / rankings.head_rank).sum() / len(rankings)
    mean_tail_r_rank = (1 / rankings.tail_rank).sum() / len(rankings)
    mrr = (mean_head_r_rank + mean_tail_r_rank) / 2

    results.append(mean_head_rank)
    results.append(mean_tail_rank)
    results.append(mean_rank)
    results.append(mean_head_r_rank)
    results.append(mean_tail_r_rank)
    results.append(mrr)

    head_hits_at_1, tail_hits_at_1, total_hits_at_1 = calc_hits_at_n(rankings, k=1)
    results.append(head_hits_at_1)
    results.append(tail_hits_at_1)
    results.append(total_hits_at_1)

    head_hits_at_3, tail_hits_at_3, total_hits_at_3 = calc_hits_at_n(rankings, k=3)
    results.append(head_hits_at_3)
    results.append(tail_hits_at_3)
    results.append(total_hits_at_3)

    head_hits_at_10, tail_hits_at_10, total_hits_at_10 = calc_hits_at_n(rankings, k=10)
    results.append(head_hits_at_10)
    results.append(tail_hits_at_10)
    results.append(total_hits_at_10)

    return pd.DataFrame([results], columns=column_headers)



