from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
import re, spacy, json, logging, multiprocessing, sys
from datetime import datetime

from nltk import word_tokenize


class TaggedDocumentIterator(object):
    def __init__(self, trainables_file):
        self.tf = trainables_file

    def __iter__(self):
        with open(self.tf, encoding='utf-8-sig') as f:
            for line in f:
                words, tags = line.split("\t")
                yield TaggedDocument(words=words.split(","), tags=tags.split(","))


class EmbeddingTrainer:
    def __init__(self):
        self.rpath = 'data/csv/data.csv'
        self.mpath = 'models/'
        self.logpath = 'log/io/'
        self.data = self.csv_to_dataframe()

    def logger(self):
        """ Sets the logger configuration to report to both std.out and to log to ./log/io/csv/
        Also sets the formatting instructions for the log file, prints Time, Current Thread, Logging Type, Message.

        """
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
            handlers=[
                logging.FileHandler("{0}/{1} ({2}).log".format(self.logpath, str(datetime.now())[:10], 'doc2vec')),
                logging.StreamHandler(sys.stdout)
            ])

    def csv_to_dataframe(self) -> pd.DataFrame:
        """Reads in CSV file declared in __init__ (self.rpath) and converts it to a number of Pandas DataFrames.

        Returns:
            pandas.DataFrame: Returns metadata CSV file as pandas DataFrame.

        """

        logger = logging.getLogger()
        data = pd.read_csv(self.rpath, delimiter=',')
        logger.info("Read file with {0} entries".format(len(data.index)))
        return data[['tag_id', 'tweet_body', 'timestamp', 'tag_sector', 'tag_industry']]

    def write_trainable_input_file(self):
        logger = logging.getLogger()
        write_path = 'data/trainable/' + str(datetime.now())[:10] + '.txt'
        logging.info("Began writing trainables file ({0})".format(write_path))
        with open(write_path, 'w', encoding='utf-8-sig') as f:
            for _, row in self.data.iterrows():
                tag_id, timestamp = row['tag_id'], row['timestamp']
                tokens = word_tokenize(row['tweet_body'])
                # tag_sector = list(map('SECTOR:{0}'.format, row['tag_sector'].split('|') if '|' in row['tag_sector'] else [row['tag_sector']]))
                # tag_industry =  list(map('INDUSTRY:{0}'.format, row['tag_industry'].split('|') if '|' in row['tag_industry'] else [row['tag_industry']]))

                words = ",".join([str(x) for x in tokens])
                # tags = ",".join([str(x) for x in [tag_id]+[timestamp]+tag_sector + tag_industry])
                line = words + "\t" + str(tag_id)
                f.write(line + "\n")
        logging.info("Finished writing trainables file ({0})".format(write_path))

    def train_model(self, tagged_docs):
        logger = logging.getLogger()
        no_of_workers = multiprocessing.cpu_count() / 2

        model = Doc2Vec(vector_size=100,
                        min_count=2,
                        dm=1,
                        workers=no_of_workers,
                        epochs=20)

        logger.info("Began building vocabulary...")
        model.build_vocab(tagged_docs, progress_per=100000)
        logger.info("Finished building vocabulary.")

        logger.info("Began training model...")
        model.train(tagged_docs,
                    total_examples=model.corpus_count,
                    epochs=model.epochs)
        logger.info("Finished training model.")

        model.save(self.mpath + str(datetime.now())[:10] + '.model')

    def run(self):
        self.logger()
        self.write_trainable_input_file()
        tagged_docs = TaggedDocumentIterator('data/trainable/2019-09-08.txt')
        self.train_model(tagged_docs)


if __name__ == "__main__":
    et = EmbeddingTrainer()
    et.run()
