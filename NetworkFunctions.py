# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 13:02:11 2025

@author: pvrika
"""

import json
import pandas as pd


def combinations(iterable, r):
    # combinations('ABCD', 2) → AB AC AD BC BD CD
    # combinations(range(4), 3) → 012 013 023 123

    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))

    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield tuple(pool[i] for i in indices)


def dist(set1: set, set2: set):
    """
    Jaccard distance between two sets is one minus the ratio between the cardinality of the intersection and the union of the sets. 

    Parameters
    ----------
    set1 : set
        A set of objects.
    set2 : set
        A set of objects.

    Returns
    -------
    float:
        Jaccard distance between set1 and set2.

    """

    try:
        pleh = len(set.union(set1, set2))

        apu = len(set.intersection(set1, set2))

        return 1-apu/pleh

    except:
        if set1 == None or set2 == None:
            return None


def distance(df, dg):

    zips = zip(map(lambda key: df.to_dict('index')[key], df.to_dict('index').keys()),
               map(lambda key: dg.to_dict('index')[key], dg.to_dict('index').keys()))

    return pd.Series(dict(zip(df.index.values, list(map(lambda x: dist(set(x[0].items()), set(x[1].items())), zips)))))


def add_data(file):
    """
    A helper to read the given datasheat. Accepts .txt, .csv, pandas.DataFrame, .json and excel.

    Parameters
    ----------
    file : string
        The path and filename of the building stock data to be analysized.

    Raises
    ------
    TypeError
        Prompts user to verify the path, filename and format of the dataset to be analysized.

    Returns
    -------
    apu2 : pandas.DataFrame
        The building stock data as a pandas.DataFrame.

    """

    if "txt" in file or "csv" in file:
        erote = ','

        apu2 = pd.read_csv(file, sep=erote, index_col=0)

    elif "json" in file:
        with open(file) as f:
            data = json.loads(f.read())
            apu2 = pd.DataFrame()
            apu2 = apu2.from_dict(data).sort_index(axis=1)

    elif type(file) == pd.DataFrame:
        apu2 = file.sort_index(axis=1)

    else:
        try:
            apu2 = pd.read_excel(file, index_col=0)
        except:
            raise TypeError(
                'File not recognized. Please check that the file is of format *.txt, *.csv, *.json or pandas.DataFrame.')

    return apu2
