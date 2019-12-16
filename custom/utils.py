from gensim.models.doc2vec import Doc2Vec
from gensim.models import Word2Vec, KeyedVectors
from pathlib import Path
from os import listdir
from os.path import isfile, join
import numpy as np
import json, re


def check_file(path, filename):
    my_file = Path(path + filename)
    try:
        my_file.resolve()
    # if my_file.is_file(): return 1
    except FileNotFoundError:
        raise


def load_model(path, model_name, type):
    check_file(path, model_name)
    if type == 'd2v':
        model = Doc2Vec.load(path + model_name)
    elif type == 'w2v':
        model = KeyedVectors.load_word2vec_format(path + model_name, binary=True)
    else:
        raise Exception('Wrong type specified')

    return model


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def lines_that_contain(string, fp):
    return [line for line in fp if string in line]


def date_prompt(self, dates):
    regexp = re.compile(r'\d{4}_\d{2}_\d{2}')

    print("\n" + "Please enter the FROM and TO dates in the format (ie. yyyy_mm_dd):")
    date_from = input("FROM> ")
    date_to = input("TO> ")

    if date_from in dates and date_to in dates and regexp.match(date_to) and regexp.match(date_from):
        if date_to > date_from:
            return date_from, date_to
        else:
            print("TO date is earlier than FROM date.")
            return 0
    else:
        print("Dates do not exist in store, please try again.")
        return 0


def count_docs_in_file(f):
    count = 0
    with open('../data/trainable/' + f) as fp:
        for line in fp:
            count += 1
    print(count)


count_docs_in_file('trainable_2017_01_01-2017_01_15.txt')
