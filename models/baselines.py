from datetime import datetime

from scipy.sparse import csr_matrix, coo_matrix, lil_matrix
from lightfm.data import Dataset
from lightfm.cross_validation import random_train_test_split as lightfm_random_train_test_split
from lightfm.evaluation import auc_score, recall_at_k, reciprocal_rank
from lightfm.evaluation import precision_at_k as lightfm_precision_at_k
from lightfm import LightFM
from collections import Counter
import numpy as np
import implicit

from spotlight.interactions import Interactions
from spotlight.cross_validation import random_train_test_split as spotlight_random_train_test_split
from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.evaluation import mrr_score, precision_recall_score, rmse_score
from spotlight.factorization.implicit import ImplicitFactorizationModel

from implicit.evaluation import train_test_split, precision_at_k
from implicit.evaluation import precision_at_k as implicit_precision_at_k

import logging
import sys
import random
import time
import os

import pandas as pd

RANDOM_STATE = np.random.RandomState(100)


class BaselineModels:
    def __init__(self):
        self.logpath = '../log/baselines/'
        self.rpath = '../data/csv/data.csv'
        self.df = self.csv_to_df()


    def logger(self, model_name):
        """Sets the logger configuration to report to both std.out and to log to ./log/models/<MODEL_NAME>/

        Also sets the formatting instructions for the log file, prints Time, Current Thread, Logging Type, Message.

        """
        directory = os.path.join(self.logpath, model_name)

        if not os.path.exists(directory):
            os.makedirs(directory)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
            handlers=[
                logging.FileHandler("{0}/{1}.log".format(directory, str(datetime.now())[:10])),
                logging.StreamHandler(sys.stdout)
            ])

    def csv_to_df(self) -> tuple:
        """Reads in CSV file, converts it to a number of Pandas DataFrames.

        Returns:
            tuple: Returns tuple of Pandas DataFrames; user features, item features and
                interactions between items.

        """

        df = pd.read_csv(self.rpath, sep=',')
        df_symbol_features = df[['user_id', 'tag_id', 'tag_industry', 'tag_sector']]
        df['count'] = df.groupby(['user_id', 'tag_id']).user_id.transform('size')
        df_weights = df[['user_id', 'tag_id', 'count']].drop_duplicates(
            subset=['user_id', 'tag_id']
        )
        df_weights['count'] = df_weights.groupby('user_id')['count'].transform(lambda x: x / x.sum())

        df = df.merge(
            df.groupby(['user_id', 'tag_id']).timestamp.agg(list).reset_index(),
            on=['user_id', 'tag_id'],
            how='left',
            suffixes=['_1', '']
        ).drop('timestamp_1', axis=1)

        df = df.groupby(
            ['user_id', 'tag_id']
        ).timestamp.agg(list).reset_index()

        listjoin = lambda x: [j for i in x for j in i]
        df['timestamp'] = df['timestamp'].apply(listjoin)
        df = df.merge(
            df_weights,
            on=['user_id', 'tag_id']
        )
        df = df.merge(
            df_symbol_features,
            on=['user_id', 'tag_id']
        )
        df = df.drop_duplicates(subset=['user_id', 'tag_id'])
        with open(os.path.join('../dataparser/', 'user_ids.txt')) as f:
            user_ids = f.readline()
            user_ids = user_ids.split(',')
            user_ids = [int(x) for x in user_ids]
        df = df[~df.user_id.isin(user_ids)]
        return df


class LightFMLib(BaselineModels):
    def __init__(self):
        super().__init__()

    def build_id_mappings(self, hybrid=False) -> Dataset:
        """Builds internal indice mapping for user-item interactions and encodes item features.

        Reads in user-item interactions and the features associated with each item and builds a mapping
        between the user and item ids from our input data to indices that will be used internally by our model.

        Item features are further encoded as an argument passed to Dataset.fit. These are supplied as a flat
        list of unique item features for the entire dataset.

        Args:
            df_interactions (pandas.DataFrame): User-Item interactions DataFrame consisting of user and item IDs.
            df_item_features (pandas.DataFrame): Item IDs and their corresponding features as column separated values.

        Returns:
            lightfm.data.Dataset: Tool for building interaction and feature matrices,
                taking care of the mapping between user/item ids and feature names and internal feature indices.
            tag_sector (list): list of all the unique cashtag sector information in the dataset.
            tag_industry (list): list of all the unique cashtag industries information in the dataset.
            :param hybrid:

        """

        dataset = Dataset()
        dataset.fit(
            (x for x in self.df['user_id']),
            (x for x in self.df['tag_id']),
            item_features=(x for x in self.df['tag_sector']) if hybrid else None
        )
        return dataset

    def build_interactions_matrix(self, dataset) -> tuple:
        """Builds a matrix of interactions between user and item.

        Takes as params a lightfm.data.Dataset object consisting of mapping between users
        and items and builds a matrix of interactions.

        Args:
            dataset (lightfm.data.Dataset): Dataset object consisting of internal user-item mappings.
            df_interactions (pandas.DataFrame): User-Item interactions DataFrame consisting of user and item IDs.

        Returns:
            tuple: Returns tuple with two scipy.sparse.coo_matrix matrices: the interactions matrix and the corresponding weights matrix.

        """

        def gen_rows(df):
            """Yields

            Args:
               df (pd.DataFrame): df_interactions matrix

            Yields:
                pandas.DataFrame: User-Item interactions DataFrame consisting of user and item IDs

            Examples:
                Generates a row, line by line of user and item IDs to pass to the lightfm.data.Dataset.build_interactions function.

                >>> print(row)
                Pandas(user_id=123456, item_id=12345678)

            """
            for row in df.itertuples(index=False):
                yield (row.user_id, row.tag_id, row.count)

        (interactions, weights) = dataset.build_interactions(gen_rows(self.df))
        interactions = interactions.tocsr().tocoo()
        return interactions, weights

    def build_item_features(self, dataset) -> csr_matrix:
        """Binds item features to item IDs, provided they exist in the fitted model.

        Takes as params a lightfm.data.Dataset object consisting of mapping between users
        and items and a pd.DataFrame object of the item IDs and their corresponding features.

        Args:
            dataset (lightfm.data.Dataset): Dataset object consisting of internal user-item mappings.
            df_item_features (pandas.DataFrame): Item IDs and their corresponding features as column separated values.

        Returns:
            scipy.sparse.csr_matrix (num items, num features): Matrix of item features.

        """

        def gen_rows(df):
            """Yields

            Args:
               df (pandas.DataFrame): df_item_features matrix

            Yields:
                pandas.DataFrame: Item IDs and their corresponding features as column separated values.

            Examples:
                Generates a row, line by line of item IDs and their corresponding features/weights to pass to the
                lightfm.data.Dataset.build_item_features function. The build_item_features function then normalises
                these weights per row.

                Also prepends each item with its type for a more accurate model.

                >>> print(row)
                [12345678, {'TAG:[CASHTAG]:2}]

            """

            for row in df.itertuples(index=False):
                yield (row.tag_id, [row.tag_sector])

            # for row in df.itertuples(index=False): d = row._asdict()
            # timestamp = 'TIME:' + str(d['timestamp'])
            # tag_sector = list(map('SECTOR:{0}'.format, d['tag_sector'].split('|') if '|' in d['tag_sector']
            # else [d['tag_sector']]))
            # tag_industry = list(map('INDUSTRY:{0}'.format,
            #                         d['tag_industry'].split('|') if '|' in d['tag_industry'] else [
            #                             d['tag_industry']]))
            #
            # weights_t = ({timestamp: 1}, Counter(tag_sector), Counter(tag_industry))
            # for weights_obj in weights_t:
            #     for k, v in weights_obj.items():
            #         yield [d['item_id'], {k: v}]

        item_features = dataset.build_item_features(gen_rows(self.df), normalize=True)
        return item_features

    def cross_validate_interactions(self, interactions) -> tuple:
        """Randomly split interactions between training and testing.

        This function takes an interaction set and splits it into two disjoint sets, a training set and a test set.

        Args:
            interactions (scipy.sparse.coo_matrix): Matrix of user-item interactions.

        Returns:
            tuple: (scipy.sparse.coo_matrix, scipy.sparse.coo_matrix), A tuple of (train data, test data).

        """

        train, test = lightfm_random_train_test_split(interactions)
        return train, test

    def evaluate_model(self, model, model_name, eval_metrics, sets, NUM_THREADS, item_features=None, k=None):
        """Evaluates models on a number of metrics

        Takes model and evaluates it depending on which evaluation metrics are passed in.
        Has local functions auc, precrec and mrr corresponding to AUC ROC score, Precision@K/Recall@K,
        Mean Reciprocal Rank metrics.

        Args:
            model (lightfm.LightFM): A LightFM model.
            model_name (str): The type of model being trained, for evaluation output purposes (Collaborative Filtering/Hybrid).
            eval_metrics (list): A list containing which evaluation metrics to carry out. Can be either of 'auc', 'precrec', 'mrr'
            sets (tuple): (scipy.sparse.coo_matrix, scipy.sparse.coo_matrix), A tuple of (train data, test data).
            NUM_THREADS (str): Number of threads to run evaluations on, corresponding to physical cores on system.
            user_features (scipy.sparse.csr_matrix, optional): Matrix of user features. Defaults to None.
            item_features (scipy.sparse.csr_matrix, optional): Matrix of item features. Defaults to None.
            k (integer, optional): The k parameter for Precision@K/Recall@K corresponding to Top-N recommendations.

        """

        logger = logging.getLogger()
        train, test = sets
        model_name = 'Collaborative Filtering' if model_name == 'cf' else 'Hybrid'

        def auc():
            """Evaluates models on the ROC AUC metric.

            Measure the ROC AUC metric for a model: the probability that a randomly chosen positive example
            has a higher score than a randomly chosen negative example. A perfect score is 1.0.

            """

            auc = auc_score(
                model=model,
                train_interactions=train,
                test_interactions=test,
                item_features=item_features
            ).mean()
            logger.info(model_name + ' AUC: %s' % auc)

            train_auc = auc_score(model,
                                  train,
                                  item_features=item_features if item_features is not None else None,
                                  num_threads=NUM_THREADS).mean()
            logger.info(model_name + ' training set AUC: %s' % train_auc)

            test_auc = auc_score(model,
                                 test,
                                 train_interactions=train,
                                 item_features=item_features if item_features is not None else None,
                                 num_threads=NUM_THREADS).mean()
            logger.info(model_name + ' test set AUC: %s' % auc)

        def precrec():
            """Evaluates models on Precision@K/Recall@K and also outputs F1 Score.

            Measure the precision at k metric for a model: the fraction of known positives in the first k
            positions of the ranked list of results. A perfect score is 1.0.

            Measure the recall at k metric for a model: the number of positive items in the first k
            positions of the ranked list of results divided by the number of positive items in the test period. #
            A perfect score is 1.0.

            Compute the F1 score, also known as balanced F-score or F-measure: The F1 score can be interpreted as a weighted
            average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0.
            The relative contribution of precision and recall to the F1 score are equal.

            """

            # train_precision = precision_at_k(model,
            #                                  train,
            #                                  k=k,
            #                                  item_features=item_features if item_features is not None else None,
            #                                  num_threads=NUM_THREADS).mean()
            # logger.info(model_name + ' training set Precision@%s: %s' % (k, train_precision))

            precision = lightfm_precision_at_k(
                model=model,
                train_interactions=train,
                test_interactions=test,
                k=k,
                item_features=item_features
            ).mean()
            logger.info(model_name + ' Precision@%s: %s' % (k, precision))

            recall = recall_at_k(
                model=model,
                train_interactions=train,
                test_interactions=test,
                k=k,
                item_features=item_features
            ).mean()
            logger.info(model_name + ' Recall@%s: %s' % (k, recall))

            fmeasure = 2 * ((precision * recall) / (precision + recall))
            logger.info(model_name + ' F-Measure: %s' % fmeasure)

        def mrr():
            """Evaluates models on their Mean Reciprocal Rank.

            Measure the reciprocal rank metric for a model: 1 / the rank of the highest ranked positive example.
            A perfect score is 1.0.

            """

            mrr = reciprocal_rank(
                model=model,
                test_interactions=test,
                train_interactions=train,
                item_features=item_features
            ).mean()
            logger.info(model_name + ' MRR: %s' % (mrr))

        for metric in eval_metrics:
            locals()[metric]()

    def run(self):
        pass


class LFMRun(LightFMLib):
    def __init__(self):
        super().__init__()
        self.filter = None
        self.loss = None
        self.model_name = 'lfm'

    def cf_model(self, train, params, item_features=None) -> LightFM:
        """Trains a pure collaborative filtering model.

        Args:
            train (scipy.sparse.coo_matrix): Training set as a COO matrix.
            params (tuple): A number of hyperparameters for the model, namely NUM_THREADS, NUM_COMPONENTS, NUM_EPOCHS, ITEM_ALPHA.

        Returns:
            lightfm.LightFM: A lightFM model.

        """

        logger = logging.getLogger()
        NUM_THREADS, NUM_COMPONENTS, NUM_EPOCHS, ITEM_ALPHA = params
        model = LightFM(
            loss=self.loss,
            item_alpha=ITEM_ALPHA,
            no_components=NUM_COMPONENTS,
            learning_rate=1e-3
        )

        logger.info('Begin fitting collaborative filtering model @ Epochs: {}'.format(NUM_EPOCHS))
        model = model.fit(
            interactions=train,
            item_features=item_features,
            epochs=NUM_EPOCHS,
            num_threads=NUM_THREADS
        )
        return model

    def run(self, filtering, loss, k):
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        self.filter = filtering
        self.loss = loss
        self.model_name = str.join('_', (self.model_name, self.filter, self.loss))

        self.logger(self.model_name)
        logger = logging.getLogger()
        logger.info("Training LightFM {0} model, Loss: {1}".format(self.filter.upper(), self.loss.upper()))
        params = (NUM_THREADS, _, _, _) = (4, 32, 10, 1e-05)

        df_interactions, df_item_features = self.df[['user_id', 'tag_id', 'count']], self.df[['timestamp']]

        dataset = self.build_id_mappings(hybrid=True if self.filter is 'hybrid' else False)
        interactions, _ = self.build_interactions_matrix(dataset)
        item_features = self.build_item_features(dataset) if self.filter is 'hybrid' else None
        train, test = self.cross_validate_interactions(interactions)

        logger.info(
            'The dataset has %s users and %s items with %s interactions in the test and %s interactions in the training set.' % (
                train.shape[0], train.shape[1], test.getnnz(), train.getnnz()))

        cf_model = self.cf_model(
            train=train,
            params=params,
            item_features=item_features
        )

        self.evaluate_model(
            model=cf_model,
            model_name=self.filter,
            eval_metrics=['mrr', 'precrec'],
            sets=(train, test),
            NUM_THREADS=NUM_THREADS,
            k=k,
            item_features=item_features
        )

        self.model_name = 'lfm'


class LFMHybrid(LightFMLib):
    def __init__(self, loss):
        super().__init__()
        self.model_name = loss+'_hybrid'

    def hybrid_model(self, params: tuple, train: coo_matrix, user_features: csr_matrix=None, item_features: csr_matrix=None) -> LightFM:
        """Trains a hybrid collaborative filtering/content model

        Adds user/item features to model to enrich training data.

        Args:
            params (tuple): A number of hyperparameters for the model, namely NUM_THREADS, NUM_COMPONENTS, NUM_EPOCHS, ITEM_ALPHA.
            train (scipy.sparse.coo_matrix): Training set as a COO matrix.
            user_features (scipy.sparse.csr_matrix) : Matrix of userfeatures.
            item_features (scipy.sparse.csr_matrix) : Matrix of item features.

        Returns:
            lightfm.LightFM: A lightFM model.

        """

        logger = logging.getLogger()
        NUM_THREADS, NUM_COMPONENTS, NUM_EPOCHS, ITEM_ALPHA = params
        model = LightFM(loss='warp',
                        item_alpha=ITEM_ALPHA,
                        no_components=NUM_COMPONENTS,
                        random_state=RANDOM_STATE)

        logger.info('Begin fitting hybrid model...')
        model = model.fit(train,
                        item_features=item_features,
                        user_features=user_features,
                        epochs=NUM_EPOCHS,
                        num_threads=NUM_THREADS)
        return model

    def run(self):
        self.logger(self.model_name)
        logger = logging.getLogger()
        params = (NUM_THREADS, _, _, _) = (4,30,10,1e-16)

        df = self.csv_to_df()
        df_interactions, df_item_features = df[['user_id', 'tag_id', 'count']], df[['timestamp','tag_industry', 'tag_sector']]

        dataset = self.build_id_mappings(df_interactions, df_item_features)
        interactions, _ = self.build_interactions_matrix(dataset, df_interactions)
        item_features = self.build_item_features(dataset, df_item_features)
        train, test = self.cross_validate_interactions(interactions)

        logger.info('The dataset has %s users and %s items with %s interactions in the test and %s interactions in the training set.' % (train.shape[0], train.shape[1], test.getnnz(), train.getnnz()))

        hybrid_model = self.hybrid_model(params, train, item_features)
        self.evaluate_model(model=hybrid_model, model_name='h', eval_metrics=['auc', 'precrec', 'mrr'], sets=(train, test), NUM_THREADS=NUM_THREADS, item_features=item_features, k=10)


class SpotlightMF(BaselineModels):
    def __init__(self):
        super().__init__()
        self.filter = None
        self.loss = None
        self.model_name = 'spot'

    def build_interactions_object(self, df_interactions: pd.DataFrame, df_timestamps: pd.DataFrame) -> Interactions:
        user_ids = df_interactions['user_id'].values.astype(int)
        cashtag_ids = df_interactions['tag_id'].values.astype(int)
        timestamps, weights = df_timestamps.values, np.array(df_interactions['count'].values)
        interactions = Interactions(
            user_ids=user_ids,
            item_ids=cashtag_ids,
            timestamps=np.array([x for x in timestamps]) if self.filter is 'hybrid' else None,
            weights=weights
        )
        return interactions

    def run(self, filtering, loss, k):
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        self.filter = filtering
        self.loss = loss
        self.model_name = str.join('_', (self.model_name, self.filter, self.loss))

        self.logger(self.model_name)
        logger = logging.getLogger()

        NUM_EPOCHS = 5

        logger.info("Training Spotlight Model, Loss: {}".format(self.loss))
        df_interactions, df_timestamps = self.df[['user_id', 'tag_id', 'count']], self.df['timestamp']
        interactions = self.build_interactions_object(df_interactions, df_timestamps)

        train, test = spotlight_random_train_test_split(interactions)
        logger.info(
            'The dataset has %s users and %s items with %s interactions in the test and %s interactions in the '
            'training set.' % (
                train.num_users, train.num_items, test.tocoo().getnnz(), train.tocoo().getnnz()))
        model = ImplicitFactorizationModel(
            n_iter=NUM_EPOCHS,
            loss=self.loss,
            random_state=RANDOM_STATE,
            use_cuda=True,
            embedding_dim=64,  # latent dimensionality
            batch_size=128,  # minibatch size
            l2=1e-9,  # strength of L2 regularization
            learning_rate=1e-3,
        )

        logger.info("Begin fitting {0} model for {1} epochs...".format(self.loss, NUM_EPOCHS))
        model.fit(train, verbose=True)

        precrec = precision_recall_score(
            model=model,
            train=train,
            test=test,
            k=k
        )

        mrr = mrr_score(
            model=model,
            train=train,
            test=test
        ).mean()

        precision = np.mean(precrec[0])
        recall = np.mean(precrec[1])
        fmeasure = 2 * ((precision * recall) / (precision + recall))
        logger.info("Precision@{0}: {1}".format(k, precision))
        logger.info("Recall@{0}: {1}".format(k, recall))
        logger.info("F-Measure: {}".format(fmeasure))
        logger.info("MRR: {}".format(mrr))
        self.model_name = 'spot'


lfm_cf = LFMRun()
spot = SpotlightMF()

for _ in range(5):
    for filtering in ['cf', 'hybrid']:
        for loss in ('bpr', 'warp', 'warp-kos'):
            lfm_cf.run(filtering, loss, 3)
        for loss in ('pointwise', 'hinge', 'bpr'):
            spot.run(filtering, loss, 3)



