import lightfm
from scipy.sparse import coo_matrix
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
from pyrecs.evaluate.mapk import mapk


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
        
    # TODO: Implement this with TFRS Streaming, in a predict module
    def predict(self, row_inds):
        # Load latent representations
        item_biases, item_factors = self.model.get_item_representations()
        user_biases, user_factors = self.model.get_user_representations()
        user_biases = user_biases[row_inds]
        user_factors = user_factors[row_inds]

        # Combine item_factors with biases for dot product
        item_factors = np.concatenate((item_factors, item_biases.reshape(-1, 1)), axis=1)

        # Add ones to user_factors for item bias
        user_factors = np.concatenate((user_factors, np.ones((user_biases.shape[0], 1))), axis=1)

        # Calculate scores
        scores = user_factors.dot(item_factors.T)

        # Sort and rank items
        n_recs = min(self.n_recs, len(item_biases))
        top_score_inds = np.argpartition(-scores, n_recs-1, axis=1)[:,:n_recs]
        sorted_top_score_inds = np.argsort(np.take_along_axis(-scores, top_score_inds, axis=1))
        top_item_inds = np.take_along_axis(top_score_inds, sorted_top_score_inds, axis=1)

        return list(zip(row_inds, top_item_inds.tolist()))

    def evaluate(self, test_dict):
        # Format truth
        truth = pd.DataFrame.from_dict(test_dict, orient='index')
        truth.index = truth.index.map(self.rows2ind)
        truth = truth.applymap(lambda x: self.columns2ind[x])
        truth = truth.T.to_dict(orient='list')

        # Predict for all train users in chunks
        num_users = len(self.rows2ind)
        row_ind_chunks = chunk_list(range(num_users), 10000)
        tqdm_desc = '[EVALUATE] Making predictions for all training users'
        predictions = list(itertools.chain(*[self.predict(ric) \
                                             for ric in tqdm(row_ind_chunks, desc=tqdm_desc, 
                                                             leave=True, position=0)]))
        predictions = {p[0]:p[1] for p in predictions}

        # Format truth and predictions for all users in truth
        # TODO: Handle new users in test that were not in train
        formatted_truth, formatted_predictions = [], []
        for user, truths in truth.items():
            formatted_predictions.append(predictions[user])
            formatted_truth.append(truths)

        return mapk(formatted_truth, formatted_predictions, k=self.n_recs)

    def train(self, model_kwargs, train_kwargs, test_dict, n_recs):
        self.n_recs = n_recs
        self.model = lightfm.LightFM(**model_kwargs)
        ones_interactions_matrix = self.interactions_matrix.copy()
        ones_interactions_matrix.data[:] = 1
        self.evaluations = []
        for i in tqdm(range(train_kwargs['num_epochs']), desc='Training', leave=True, position=0):
            self.model.fit_partial(interactions=ones_interactions_matrix, 
                                   user_features=train_kwargs['user_features'], 
                                   item_features=train_kwargs['item_features'], 
                                   sample_weight=self.interactions_matrix, 
                                   epochs=1, num_threads=train_kwargs['num_threads'], 
                                   verbose=False)
            self.evaluations.append(self.evaluate(test_dict))
        print(self.evaluations)
    
    def run(self, flat_interactions_df, lightfm_model, train_kwargs):
        self.preprocess(flat_interactions_df)
        self.train(lightfm_model, train_kwargs, test_dict)