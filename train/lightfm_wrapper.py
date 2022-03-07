import lightfm
from scipy.sparse import coo_matrix
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
from pyrecs.evaluate.mapk import mapk
from pyrecs.predict import tfrs_streaming


def chunk_list(lst, n):
        res = []
        for i in range(0, len(lst), n):
            res.append(lst[i:i + n])
        return np.array(res)

class LightFM:
    def __init__(self, rows_col, columns_col, interactions_type):
        self.rows_col = rows_col
        self.columns_col = columns_col
        self.interactions_type = interactions_type
        
    def preprocess(self, flat_interactions_df):
        self.rows2ind = dict(zip(flat_interactions_df[self.rows_col].unique(),
                                 range(flat_interactions_df[self.rows_col].nunique())))
        self.columns2ind = dict(zip(flat_interactions_df[self.columns_col].unique(),
                                    range(flat_interactions_df[self.columns_col].nunique())))
        if self.interactions_type == 'ones':
            flat_interactions_df.drop_duplicates(subset=[self.rows_col, self.columns_col], inplace=True)
            rows = flat_interactions_df[self.rows_col].map(self.rows2ind).values
            cols = flat_interactions_df[self.columns_col].map(self.columns2ind).values
            interactions = np.ones(len(flat_interactions_df))
        elif self.interactions_type == 'counts':
            counts_dict = dict(flat_interactions_df.groupby([self.rows_col, self.columns_col]).size())
            unique_row_column_pairs = flat_interactions_df.groupby('users')['items'].unique().reset_index().explode('items').values
            rows, cols, interactions = [], [], []
            for unique_pair in unique_row_column_pairs:
                row, column = unique_pair
                rows.append(self.rows2ind[row])
                cols.append(self.columns2ind[column])
                interactions.append(counts_dict[(row, column)])
        else:
            raise ValueError(f"interactions_type ('{self.interactions_type}') not implemented.")
        self.interactions_matrix = coo_matrix((interactions, (rows, cols)), 
                                              shape=(len(self.rows2ind), len(self.columns2ind)))

    # TODO: add user/item feature functionality, and evaluation
    def evaluate(self, test_df):
        # Format truth
        truth = dict(test_df.groupby(self.rows_col)[self.columns_col].apply(lambda x: list(x.unique())))
        
        # Predict for all train users
        item_biases, item_factors = self.model.get_item_representations()
        user_biases, user_factors = self.model.get_user_representations()
        item_factors = np.concatenate((item_factors, item_biases.reshape(-1, 1)), axis=1)
        user_factors = np.concatenate((user_factors, np.ones((user_biases.shape[0], 1))), axis=1)
        user_identifiers = [r[0] for r in sorted(self.rows2ind.items(), key=lambda x: x[1])]
        item_identifiers = [c[0] for c in sorted(self.columns2ind.items(), key=lambda x: x[1])]
        predictions = tfrs_streaming.predict(user_identifiers=user_identifiers, 
                                             user_embeddings=user_factors,
                                             item_identifiers=item_identifiers, 
                                             item_embeddings=item_factors,
                                             embedding_dtype='float32', 
                                             n_recs=self.n_recs,
                                             prediction_batch_size=self.tfrs_prediction_batch_size)

        # Format truth and predictions for all users in truth
        # TODO: For new users, predict the most popular items
        # TODO: Add functionality to filter out items that existing users have already interacted with
        formatted_truth, formatted_predictions = [], []
        for user, truths in truth.items():
            formatted_predictions.append(predictions.get(user, []))
            formatted_truth.append(truths)

        return mapk(formatted_truth, formatted_predictions, k=self.n_recs)

    def train(self, model_kwargs, train_kwargs, test_df, n_recs, tfrs_prediction_batch_size):
        self.n_recs = n_recs
        self.tfrs_prediction_batch_size = tfrs_prediction_batch_size
        self.model = lightfm.LightFM(**model_kwargs)
        ones_interactions_matrix = self.interactions_matrix.copy()
        ones_interactions_matrix.data[:] = 1
        self.test_set_evaluations = []
        # TODO: Optional train/test set evaluations
        # TODO: Evalulation on some epochs only
        for i in tqdm(range(train_kwargs['num_epochs']), desc='Training', leave=True, position=0):
            self.model.fit_partial(interactions=ones_interactions_matrix, 
                                   user_features=train_kwargs['user_features'], 
                                   item_features=train_kwargs['item_features'], 
                                   sample_weight=self.interactions_matrix, 
                                   epochs=1, num_threads=train_kwargs['num_threads'], 
                                   verbose=False)
            self.test_set_evaluations.append(self.evaluate(test_df))
        plt.plot(range(1, train_kwargs['num_epochs']+1), self.test_set_evaluations)
        plt.xticks(range(1, train_kwargs['num_epochs']+1))
        plt.title('Test Set Performance')
        plt.xlabel('Epoch')
        plt.ylabel(f'MAP@{self.n_recs}')
    
    def run(self, flat_interactions_df, lightfm_model, train_kwargs, test_df):
        self.preprocess(flat_interactions_df)
        self.train(lightfm_model, train_kwargs, test_df)