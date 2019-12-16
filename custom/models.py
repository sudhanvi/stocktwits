import json
import logging
import multiprocessing
import sys
import time
from datetime import timedelta, datetime
from os import listdir, makedirs
from os.path import isfile, join, exists

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from custom.analysis import Doc2VecAnalysis, Analysis
from custom.utils import check_file, NumpyEncoder, load_model

""" DOC2VEC RELATED FUNCTIONALITY
"""


class TaggedDocumentIterator(object):
    def __init__(self, doc):
        self.path_data = '../data/'
        self.trainables_path = self.path_data + 'trainable/'
        self.doc = doc

    def __iter__(self):
        with open(self.trainables_path + self.doc, encoding='utf-8-sig') as f:
            for line in f:
                s_line = line.strip('\n')
                tokens = s_line.split(" ")
                tag = tokens[0]
                words = tokens[1:]
                yield TaggedDocument(words=words, tags=[tag])


class D2VTraining:
    def __init__(self, model_name, vec_size, epochs):
        self.path_data = '../data/'
        self.tweets_path = self.path_data + 'tweets/'
        self.trainables_path = self.path_data + 'trainable/'
        self.log_path = '../log/'
        self.path_models = '../models/'
        self.model_name = model_name
        self.path_models_subpath = self.path_models+self.model_name[:-6]+'/'
        self.epochs = epochs
        self.vec_size = vec_size

    def query_trainable(self):
        title = "TRAIN MODEL (DOC2VEC: "
        store = [f[10:-4] for f in listdir(self.trainables_path) if
                 isfile(join(self.trainables_path, f)) and "trainable_" in f]
        store.sort()

        print("\n" + title + self.model_name + ")\n" + ("-" * (
                len(title) + len(self.model_name) + 1)) + "\n" + "The following trainable files are available:",
              end="\n")
        print(store)

        print("\nPlease choose a corresponding file:")
        trainable_file = input("> ")

        while trainable_file not in store:
            print("\nNot in store, please try again.")
            trainable_file = input("> ")

        return "trainable_" + trainable_file + ".txt"

    def train_model(self, tagged_docs):
        max_epochs = self.epochs
        vec_size = self.vec_size
        no_of_workers = multiprocessing.cpu_count() / 2
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
            handlers=[
                logging.FileHandler("{0}/{1}_log ({2}).log".format(self.log_path, str(datetime.now())[:10], self.model_name)[:-6]),
                logging.StreamHandler(sys.stdout)
            ])
        # alpha = 0.025

        model = Doc2Vec(vector_size=vec_size,
                        min_count=2,
                        dm=1,
                        workers=no_of_workers,
                        epochs=max_epochs)

        # BUILD VOCABULARY
        print("\nBuilding vocabulary started:", str(datetime.now()))
        logging.info('.. Build vocabulary ' + str(datetime.now()))
        vocab_start_time = time.monotonic()

        model.build_vocab(tagged_docs, progress_per=250000)

        vocab_end_time = time.monotonic()
        print("Building vocabulary ended:", str(datetime.now()) + ".", "Time taken:",
              timedelta(seconds=vocab_end_time - vocab_start_time), "Size:", len(model.wv.vocab))
        logging.info('.. Build vocabulary ended ' + str(datetime.now()) + " Time taken: " + str(
            timedelta(seconds=vocab_end_time - vocab_start_time)) + "Size: " + str(len(model.wv.vocab)))

        # TRAIN MODEL
        print("Training began:", str(datetime.now()), "Vector Size:", vec_size, "Epochs:", max_epochs)
        logging.info(
            '.. Train model ' + str(datetime.now()) + "Vector Size: " + str(vec_size) + ", Epochs:" + str(max_epochs))
        start_time = time.monotonic()

        model.train(tagged_docs,
                    total_examples=model.corpus_count,
                    epochs=model.epochs)

        end_time = time.monotonic()
        print("Training Ended:", str(datetime.now()) + ".", "Time taken:",
              str(timedelta(seconds=end_time - start_time)))
        logging.info('.. Train model ended ' + str(datetime.now()) + ' Time taken: ' + str(
            timedelta(seconds=end_time - start_time)))

        # SAVE MODEL
        if not exists(self.path_models + self.model_name[:-6] + '/'):
            makedirs(self.path_models + self.model_name[:-6] + '/')
        model.save(self.path_models + self.model_name[:-6] + '/' + self.model_name)
        print("Model Saved")

        # CLEAR LOG CONFIGS
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)


class D2VModel:
    def __init__(self, model):
        self.path_data = '../data/'
        self.path_models = '../models/'
        self.model = load_model(self.path_models, model, 'd2v')

    """ Functions below listed for writing to files.
    """

    def save_user_embeddings(self, obj):
        if len(list(obj.keys())) > 1:
            print("Writing to file...")
            with open(self.path_data + obj['ue_store_filename'][:-8] + "_EMBD.txt", 'w') as fp:
                for k, v in obj.items():
                    json.dump({k: v}, fp, cls=NumpyEncoder)
                    fp.write("\n")
            print("Write complete")
        else:
            print("Empty embeddings object passed as argument")

    """ Functions below are for user embedding calculation.
        build_user_embeddings_store calculates an average of all of the document vectors
        for a particular user, builds them to a store which can then be written to a file.
    """

    def get_user_embedding(self, user):
        user_doc_vecs = [self.model.docvecs[tag] for tag in self.model.docvecs.offset2doctag if tag.startswith(user)]
        user_vec = sum(user_doc_vecs) / len(user_doc_vecs)
        return user_vec

    def build_user_embeddings_store(self, filename):
        user_embeddings = {'ue_store_filename': filename}
        check_file(self.path_data, filename)
        print("User embeddings process began:", str(datetime.now()))
        start_time = time.monotonic()
        with open(self.path_data + filename) as f:
            for line in f:
                user = list(json.loads(line).keys())[0]
                user_vec = self.get_user_embedding(user)
                user_embeddings[user] = user_vec
        end_time = time.monotonic()
        print("User embeddings process ended:", str(datetime.now()) + ".", "Time taken:",
              timedelta(seconds=end_time - start_time))
        return user_embeddings


""" WORD2VEC RELATED FUNCTIONALITY
"""


class W2VModel:
    def __init__(self, model):
        self.model_path = '../models/'
        self.model_name = model
        self.model = load_model(self.model_path, self.model_name, 'w2v')


if __name__ == "__main__":
    # t = D2VTraining(model_name='d2v_100d_10e_dm_2017_q12.model', vec_size=100, epochs=10)
    # tagged_docs = TaggedDocumentIterator(t.query_trainable())
    # t.train_model(tagged_docs)

    model = D2VModel
    analysis = Doc2VecAnalysis('2019-09-08', model, 'd2v')
    analysis.log()
