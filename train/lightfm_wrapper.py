import lightfm
from scipy.sparse import coo_matrix
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
from pyrecs.evaluate.mapk import mapk
from pyrecs.predict import tfrs_streaming


class LightFM:
    def __init__(self, users_col, items_col, interactions_type, 
                 model_kwargs, train_kwargs, 
                 n_recs, tfrs_prediction_batch_size):
        self.users_col = users_col
        self.items_col = items_col
        self.interactions_type = interactions_type
        self.model_kwargs = model_kwargs
        self.train_kwargs = train_kwargs
        self.n_recs = n_recs
        self.tfrs_prediction_batch_size = tfrs_prediction_batch_size
        
    def preprocess(self, flat_interactions_df, test_df):
        # Format truth
        self.truth = dict(test_df.groupby(self.users_col)[self.items_col].apply(lambda x: list(x.unique())))
        
        # Create training interactions
        self.user2ind = dict(zip(flat_interactions_df[self.users_col].unique(),
                                 range(flat_interactions_df[self.users_col].nunique())))
        self.item2ind = dict(zip(flat_interactions_df[self.items_col].unique(),
                                    range(flat_interactions_df[self.items_col].nunique())))
        if self.interactions_type == 'ones':
            flat_interactions_df.drop_duplicates(subset=[self.users_col, self.items_col], inplace=True)
            users = flat_interactions_df[self.users_col].map(self.user2ind).values
            cols = flat_interactions_df[self.items_col].map(self.item2ind).values
            interactions = np.ones(len(flat_interactions_df))
        elif self.interactions_type == 'counts':
            counts_dict = dict(flat_interactions_df.groupby([self.users_col, self.items_col]).size())
            unique_user_item_pairs = flat_interactions_df.groupby(self.users_col)[self.items_col].unique()\
                                            .reset_index().explode(self.items_col).values
            users, cols, interactions = [], [], []
            for unique_pair in unique_user_item_pairs:
                user, item = unique_pair
                users.append(self.user2ind[user])
                cols.append(self.item2ind[item])
                interactions.append(counts_dict[(user, item)])
        else:
            raise ValueError(f"interactions_type ('{self.interactions_type}') not implemented.")
        self.interactions_matrix = coo_matrix((interactions, (users, cols)), 
                                              shape=(len(self.user2ind), len(self.item2ind)))
        
        # Make sure n_recs isn't > number of items available
        self.n_recs = min(self.n_recs, len(self.item2ind))

    # TODO: add user/item feature functionality, and evaluation
    def evaluate(self):        
        # Predict for all train users
        item_biases, item_factors = self.model.get_item_representations()
        user_biases, user_factors = self.model.get_user_representations()
        item_factors = np.concatenate((item_factors, item_biases.reshape(-1, 1)), axis=1)
        user_factors = np.concatenate((user_factors, np.ones((user_biases.shape[0], 1))), axis=1)
        user_identifiers = [r[0] for r in sorted(self.user2ind.items(), key=lambda x: x[1])]
        item_identifiers = [c[0] for c in sorted(self.item2ind.items(), key=lambda x: x[1])]
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
        for user, truths in self.truth.items():
            formatted_predictions.append(predictions.get(user, []))
            formatted_truth.append(truths)

        return mapk(formatted_truth, formatted_predictions, k=self.n_recs)

    def train(self):
        self.model = lightfm.LightFM(**self.model_kwargs)
        ones_interactions_matrix = self.interactions_matrix.copy()
        ones_interactions_matrix.data[:] = 1
        self.test_set_evaluations, self.test_set_eval_epochs = [], []
        # TODO: Train set evaluation
        for epoch in tqdm(range(1, self.train_kwargs['num_epochs']+1), desc='Training', leave=True, position=0):
            self.model.fit_partial(interactions=ones_interactions_matrix, 
                                   user_features=self.train_kwargs['user_features'], 
                                   item_features=self.train_kwargs['item_features'], 
                                   sample_weight=self.interactions_matrix, 
                                   epochs=1, num_threads=self.train_kwargs['num_threads'], 
                                   verbose=False)
            if (self.train_kwargs['test_eval_epochs'] == 'all') or (epoch in self.train_kwargs['test_eval_epochs']):
                self.test_set_eval_epochs.append(epoch)
                self.test_set_evaluations.append(self.evaluate())
        plt.plot(self.test_set_eval_epochs, self.test_set_evaluations, linestyle='dotted')
        plt.scatter(self.test_set_eval_epochs, self.test_set_evaluations, label='Test Set')
        plt.xticks(self.test_set_eval_epochs)
        plt.legend()
        plt.title('Evaluation')
        plt.xlabel('Epoch')
        plt.ylabel(f'MAP@{self.n_recs}')
    
    def run(self, flat_interactions_df, test_df):
        self.preprocess(flat_interactions_df, test_df)
        self.train()