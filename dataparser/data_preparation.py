from datetime import datetime
from sklearn.utils import shuffle
from sklearn.datasets import dump_svmlight_file
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

import scipy.sparse

import os
import logging
import sys
import json
import re
import time


def clear_logger_settings():
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)


class DataPrep:
    def __init__(self, name):
        self.st_path = 'H:/dissertation/st2017'
        self.dpath = '../data/csv/'
        self.logpath = '../log/csv/dataparser'
        self.name = name

    def logger(self):
        """ Sets the logger configuration to report to both std.out and to log to ./log/io/csv/
        Also sets the formatting instructions for the log file, prints Time, Current Thread,
        Logging Type, Message.

        """
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
            handlers=[
                logging.FileHandler("{0}/{1}_log ({2}).log".format(self.logpath, str(datetime.now())[:10], self.name)),
                logging.StreamHandler(sys.stdout)
            ])


class DataParser(DataPrep):
    def __init__(self, limit: str = None):
        super().__init__('data_parser')

        self.files = [f for f in os.listdir(self.st_path) if os.path.isfile(os.path.join(self.st_path, f))]
        self.files.sort()
        ans = []
        for x in self.files:  # iterate over it, using x as variable
            if x != '/':  # if the element is not 'x'
                ans.append(x[1:])

    def get_symbols_only(self):
        logger = logging.getLogger()
        rows = []

        for i, fp in enumerate(self.files):
            logger.info("Parsing file <{}>, {}/{}".format(fp, i + 1, len(self.files)))
            with open(os.path.join(self.st_path, fp), encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)['data']
                    symbols = data.get('symbols')
                    if not symbols:
                        continue

                    for s in symbols:
                        check = (s_id, s_industry, s_sector, s_title, s_exchange) = s.get('id'), s.get(
                            'industry'), s.get(
                            'sector'), s.get('title'), s.get('exchange')
                        if not all(check):
                            continue

                        s_industry = re.sub('[^\w]+', '', s_industry)
                        s_sector = re.sub('[^\w]+', '', s_sector)
                        s_title = re.sub('[^\w]+', '', s_title)
                        s_exchange = re.sub('[^\w]+', '', s_exchange)

                        timestamp = data.get('created_at')
                        timestamp = re.sub('T|Z', ' ', timestamp).strip()
                        timestamp = int(time.mktime(time.strptime(timestamp, '%Y-%m-%d %H:%M:%S')))

                        tweet_body = data.get('body').replace('\t', ' ').replace('\n', '')

                        rows.append({
                            'user_id': data['user']['id'],
                            'tag_id': s_id,
                            'timestamp': timestamp,
                            'tweet_body': tweet_body,
                            'tag_industry': s_industry,
                            'tag_sector': s_sector,
                            'title': s_title,
                            'exchange': s_exchange
                        })

        df = pd.DataFrame(rows, columns=['user_id', 'tag_id', 'timestamp', 'tweet_body', 'tag_industry', 'tag_sector',
                                         'title', 'exchange'])

        outpath = os.path.join(self.dpath, 'symbols.csv')
        df.to_csv(outpath, sep=',', index=False)

    def run(self):
        self.logger()
        self.get_symbols_only()
        clear_logger_settings()


class DataCleaner(DataPrep):
    def __init__(self, fp):
        super().__init__('data_cleaner')
        self.df = pd.read_csv(os.path.join(self.dpath, fp), sep=',')

    def format_data(self):
        logger = logging.getLogger()
        logger.info("Begin data formatting... Dataframe Size: {}".format(self.df.shape[0]))

        # tag_features = self.df[['tag_id', 'tag_industry', 'tag_sector', 'title', 'exchange']]
        # self.df = self.df.merge(
        #     self.df.groupby(['user_id', 'tag_id']).timestamp.agg(list).reset_index(),
        #     on=['user_id', 'tag_id'],
        #     how='left',
        #     suffixes=['_1', '']
        # ).drop('timestamp_1', axis=1)
        # self.df['timestamp'] = self.df['timestamp'].apply(lambda x: x[-1])
        # counts = self.df.groupby(['user_id', 'tag_id']).size().reset_index().rename(columns={0: 'count'})
        # self.df = self.df.merge(
        #     counts,
        #     on=['user_id', 'tag_id']
        # )
        # self.df = self.df.drop_duplicates(subset=['user_id', 'tag_id'])
        #
        # logger.info("Saving interactions... Dataframe Size: {}".format(self.df.shape[0]))
        # outpath = os.path.join(self.dpath, '02_interactions.csv')
        # self.df.to_csv(outpath, sep=',', index=False)
        # sys.exit(0)

    def bot_cleaner(self):
        logger = logging.getLogger()
        bot_ids = [47688, 2843, 74023, 356080, 348830, 373849, 406225, 730219, 894342, 347689]

        data_count = self.df.shape[0]
        logger.info("Begin bot removal... Dataframe Size: {}".format(data_count))
        self.df = self.df[~self.df.user_id.isin(bot_ids)]
        logger.info("Removed bots. Size of DataFrame: {0} -> {1}".format(data_count, self.df.shape[0]))

        outpath = os.path.join(self.dpath, 'data.csv')
        self.df.to_csv(outpath, sep=',', index=False)

    def interaction_threshold(self, k):
        logger = logging.getLogger()

        data_count = self.df.shape[0]
        logger.info("Begin cleaning... Dataframe Size: {}".format(data_count))
        cold_start_users = self.df.groupby(['user_id', 'tag_id']).filter(lambda x: len(x) < k)['user_id'].values
        self.df = self.df[~self.df.user_id.isin(cold_start_users)]
        self.df['target'] = self.df[['tag_industry', 'tag_sector']].apply(lambda x: '|'.join(x), axis=1)
        logger.info(
            "Removed users with less than {0} unique interactions. Size of DataFrame: {1} -> {2}".format(k, data_count,
                                                                                                         self.df.shape[
                                                                                                             0]))

        outpath = os.path.join(self.dpath, 'data.csv')
        self.df.to_csv(outpath, sep=',', index=False)

    def run(self, k):
        self.logger()

        self.format_data()
        self.bot_cleaner()
        self.interaction_threshold(k)

        clear_logger_settings()


class LibSVMParser(DataPrep):
    def __init__(self, fp):
        super().__init__('libsvmparser')
        self.df = pd.read_csv(os.path.join(self.dpath, fp), sep=',')
        # self.df = self.df.drop('timestamp', axis=1)
        self.df['tg'] = 1
        self.df = self.df.drop_duplicates(subset=['user_id', 'tag_id'])

    def categorise_features(self):
        self.df.tag_sector = pd.Categorical(self.df.tag_sector)
        self.df.tag_industry = pd.Categorical(self.df.tag_industry)
        self.df['tag_sector'] = self.df.tag_sector.cat.codes
        self.df['tag_industry'] = self.df.tag_industry.cat.codes

        user_tags = self.df.groupby('user_id')['tag_id'].apply(list)
        user_sectors = self.df.groupby('user_id')['tag_sector'].apply(list)
        user_industries = self.df.groupby('user_id')['tag_industry'].apply(list)

        self.df['user_tags'] = ''
        self.df['user_sectors'] = ''
        self.df['user_industries'] = ''

        for i, row in self.df.iterrows():
            user_id = row['user_id']
            self.df.at[i, 'user_tags'] = user_tags[user_id]
            self.df.at[i, 'user_sectors'] = user_sectors[user_id]
            self.df.at[i, 'user_industries'] = user_industries[user_id]

        listjoin = lambda x: ','.join(map(str, x))
        self.df['user_tags'] = self.df['user_tags'].apply(listjoin)
        self.df['user_sectors'] = self.df['user_sectors'].apply(listjoin)
        self.df['user_industries'] = self.df['user_industries'].apply(listjoin)

    def neg_sampling(self, df):
        def remove_potential_bots(dframe):
            user_sel = None
            user_sel_tags = []

            for _, row in df.iterrows():
                user_sel_tags = df.tag_id[df.user_id == row.user_id].tolist()
                try:
                    neg_sample = df[~df.tag_id.isin(user_sel_tags)].sample(n=2)
                except ValueError:
                    print("Removed potential bot, UserID: {}. Logging...".format(row.user_id))
                    return (row.user_id, dframe)
                if user_sel != row.user_id:
                    user_sel = row.user_id
                    user_sel_tags = dframe.tag_id[df.user_id == user_sel].tolist()
                    neg_sample = dframe[~dframe.tag_id.isin(user_sel_tags)].sample(n=2)
                    neg_tags = neg_sample.tag_id.tolist()
                    user_sel_tags += neg_tags
                else:
                    try:
                        neg_sample = dframe[~dframe.tag_id.isin(user_sel_tags)].sample(n=2)
                        neg_tags = neg_sample.tag_id.tolist()
                        user_sel_tags += neg_tags
                    except ValueError:
                        print("Removed potential bot, UserID: {}. Logging...".format(row.user_id))
                        return (row.user_id, dframe)
            return dframe

        print("Dataframe entries: {}, Beginning bot removal...".format(df.shape[0]))
        bot_entries = remove_potential_bots(df)
        while isinstance(bot_entries, tuple):
            bot_user, df = bot_entries
            df = df[df.user_id != bot_user]
            print("Updated DataFrame, now contains {} entries".format(df.shape[0]))
            bot_entries = remove_potential_bots(df[df.user_id != bot_user])
        df = bot_entries
        print("Bot removal completed. Dataframe entries: {}".format(df.shape[0]))
        df = df.reset_index(drop=True)
        df.to_csv(self.dpath + 'fm_bot_clean.csv', sep=",", index=False)
        # sys.exit(0)

        user_sel = None
        user_sel_tags = []
        init_shape = df.shape[0]

        to_drop = []

        for i, row in df.iterrows():
            print("{}/{}".format(i + 1, init_shape))
            if row.tg == -1: continue
            if user_sel != row.user_id:
                user_sel = row.user_id
                user_sel_tags = df.tag_id[df.user_id == user_sel].tolist()

                try:
                    neg_sample = df[~df.tag_id.isin(user_sel_tags)].sample(n=2)
                except ValueError:
                    to_drop.append(row.user_id)
                    continue
                neg_sample.user_id = row.user_id
                neg_sample.tg = -1
                neg_tags = neg_sample.tag_id.tolist()
                df = df.append(neg_sample)
                # user_sel_tags += neg_tags
            else:
                try:
                    neg_sample = df[~df.tag_id.isin(user_sel_tags)].sample(n=2)
                except ValueError:
                    to_drop.append(row.user_id)
                    continue
                neg_sample.user_id = row.user_id
                neg_sample.tg = -1
                neg_tags = neg_sample.tag_id.tolist()
                df = df.append(neg_sample)
                # user_sel_tags += neg_tags

        print("Bots dropped: {}".format(len(to_drop)))
        df = df[~df.user_id.isin(to_drop)]
        pos_samples, neg_samples = df[df.tg == 1].shape[0], df[df.tg == -1].shape[0]
        print("Positive samples = {}, Negative samples = {}".format(pos_samples, neg_samples))
        df = shuffle(df)
        df = df.reset_index(drop=True)
        return df

    def split_train_test(self, sparse_matrices, validation=False) -> tuple:
        X, y = sparse_matrices
        if validation:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=1)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.25, random_state=1)

            return (
                (X_train, y_train),
                (X_val, y_val),
                (X_test, y_test)
            )

        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.33, random_state=5)

            return (
                (X_train, y_train),
                (X_test, y_test)
            )
        raise Exception("Split Error")

    def dummify(self, ret_t):
        df = pd.read_csv(os.path.join(self.dpath, 'libsvm_bots.csv'), sep=',')
        print(df.shape[0])
        user_ids = df.copy()
        user_ids = user_ids.drop_duplicates(subset='user_id')
        user_ids = user_ids.sample(frac=0.75)
        user_ids = user_ids.user_id.tolist()
        with open('user_ids.txt', 'w') as f:
            f.write(','.join([str(x) for x in user_ids]))
        # sys.exit(0)
        with open('user_ids.txt') as f:
            user_ids = f.readline()
            user_ids = user_ids.split(',')
            user_ids = [int(x) for x in user_ids]
        df = df[~df.user_id.isin(user_ids)]
        print(df.shape[0])

        cols = list(df)
        cols.insert(0, cols.pop(cols.index('tg')))
        df = df[cols]

        user_tags = df.pop('user_tags')
        user_sectors = df.pop('user_sectors')
        user_industries = df.pop('user_industries')

        cols = list(df)[1:]

        u_tags_dummies = user_tags.str.get_dummies(sep=',').astype(np.int8).add_prefix('user_tags_')
        u_sectors_dummies = user_sectors.str.get_dummies(sep=',').astype(np.int8).add_prefix('user_sectors_')
        u_industries_dummies = user_industries.str.get_dummies(sep=',').astype(np.int8).add_prefix('user_industries_')

        u_tags_dummies = u_tags_dummies.div(u_tags_dummies.sum(axis=1), axis=0).round(4)
        u_sectors_dummies = u_sectors_dummies.div(u_sectors_dummies.sum(axis=1), axis=0).round(4)
        u_industries_dummies = u_industries_dummies.div(u_industries_dummies.sum(axis=1), axis=0).round(4)

        X = pd.get_dummies(data=df, columns=cols)

        X = pd.concat(
            [X, u_tags_dummies, u_sectors_dummies, u_industries_dummies],
            axis=1
        )

        y = X.pop('tg')
        X, y = X.astype('int32'), y.astype('int32')
        print(X.shape, y.shape)

        return_type = {
            # 'df': (df, y.transpose()),
            # 'dense': (np.array(X), np.transpose(np.array(y))),
            'sparse': (scipy.sparse.csr_matrix(X.values).astype(float),
                       scipy.sparse.csr_matrix(y.values).transpose().astype(float))
        }
        return return_type[ret_t]

    def run(self):
        # self.categorise_features()
        # self.df = self.neg_sampling(self.df)
        # self.df.to_csv(os.path.join(self.dpath, 'libsvm_bots.csv'), sep=',', index=False)
        aliases = ['train', 'test']
        dummies = self.dummify('sparse')
        splits = self.split_train_test(dummies, validation=True)
        if len(splits) == 3:
            aliases = aliases[:1] + ['validation'] + aliases[1:]
        for i, s in enumerate(splits):
            X, y = s
            print("Writing {} file".format(aliases[i]))
            dump_svmlight_file(X, y, os.path.join(self.dpath, aliases[i] + '.libfm'))


# dp = DataParser('2017_01_15')
# dp.run()
# dc = DataCleaner('symbols.csv')
# dc.run(5)

libsvm = LibSVMParser('libsvm_bots.csv')
libsvm.run()
