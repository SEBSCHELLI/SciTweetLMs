import os
from functools import partial
from itertools import chain
from typing import Dict, Union, Any, List, Tuple
import ast
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from sklearn.model_selection import KFold, StratifiedKFold
from src.preprocess_tweets import preprocess_tweets

if 'SciTweetLMs' not in os.getcwd():
    os.chdir('/home/schellsn/SciTweetLMs') #Change

class LMDataLoader:
    @staticmethod
    def get_train_and_test_data(path, tokenizer, preprocessing_config, split, seed):

        def tokenize(examples):
            return tokenizer(examples["text"], max_length=128, truncation=True, padding='max_length')

        data = pd.read_pickle(path)

        data = data[['tweetId', 'text', 'url', '']]

        data = preprocess_tweets(data, **preprocessing_config)

        train_data = data.sample(frac=split, random_state=seed)
        test_data = data[~data.index.isin(train_data.index)]

        assert len(train_data[train_data.index.isin(test_data.index)]) == 0

        train_data = Dataset.from_pandas(train_data)
        train_data = train_data.map(tokenize, batched=True, batch_size=8)

        test_data = Dataset.from_pandas(test_data)
        test_data = test_data.map(tokenize, batched=True, batch_size=8)

        return train_data, test_data


class SciTweetsDataLoader:
    def __init__(self, cat, tokenizer, n_folds, preprocessing_config, seed=0):
        self.cat = cat
        self.tokenizer = tokenizer
        self.n_folds = n_folds
        self.preprocessing_config = preprocessing_config
        self.seed = seed
        self.data = self._read_data()
        self.cv_datasets, self.cv_test_dfs = self._create_datasets()

    def _read_data(self):
        data = pd.read_csv('eval_data/scitweets/annotations.tsv', sep='\t')
        data['processed_urls'] = data['processed_urls'].apply(lambda urls: ast.literal_eval(urls))
        data['original_text'] = data['text']

        if self.cat in ['cat1', 'cat2', 'cat3']:
            data['labels'] = data[f'{self.cat}_final_answer']

        elif self.cat == "scirelated":
            print('not implemented yet. Using multilabel instead')
            self.cat = "multilabel"

        elif self.cat == "multilabel":
            data['labels'] = data[['cat1_final_answer', 'cat2_final_answer', 'cat3_final_answer']].apply(lambda x: [x[0], x[1], x[2]], axis=1)

        data = data[~data['labels'].astype(str).str.contains("0.5")]

        data = preprocess_tweets(data, **self.preprocessing_config)

        return data

    def _create_datasets(self):

        def tokenize(examples):
            return self.tokenizer(examples["text"], max_length=128, truncation=True, padding='max_length')

        def filter_split(split_indices: List, example: Union[Dict, Any], indices: int) -> List[bool]:
            return [True if idx in split_indices else False for idx in indices]

        cv_datasets = []
        cv_test_dfs = []

        dataset = Dataset.from_pandas(self.data[['text', 'labels']])
        dataset = dataset.map(tokenize, batched=True, batch_size=32, remove_columns=['text', '__index_level_0__'])

        kf = KFold(n_splits=self.n_folds, random_state=self.seed, shuffle=True)
        kf.get_n_splits(self.data)

        for fold, (train_index, test_index) in enumerate(kf.split(data)):
            assert len(set(train_index).intersection(set(test_index))) == 0

            train_dataset = dataset.filter(partial(filter_split, train_index), with_indices=True, batched=True, keep_in_memory=True)
            test_dataset = dataset.filter(partial(filter_split, test_index), with_indices=True, batched=True, keep_in_memory=True)

            cv_datasets.append((train_dataset, test_dataset))
            cv_test_dfs.append(self.data.iloc[test_index])

        return cv_datasets, cv_test_dfs

    def get_datasets_for_fold(self, fold):
        return self.cv_datasets[fold]

    def get_test_df_for_fold(self, fold):
        return self.cv_test_dfs[fold]
