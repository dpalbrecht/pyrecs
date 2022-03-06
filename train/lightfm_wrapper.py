import lightfm
from scipy.sparse import coo_matrix
import numpy as np
from tqdm import tqdm
from pyrecs.evaluate import mapk

class LightFM:
    def __init__(self, rows_col, columns_col, interactions_type):
        self.rows_col = rows_col
        self.columns_col = columns_col
        self.interactions_type = interactions_type
        
    def _make_interactions_matrix(self, flat_interactions_df):
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
    
    def preprocess(self, flat_interactions_df):
        self._make_interactions_matrix(flat_interactions_df)
        
    def evaluate(self):
        # TODO: 
        pass
    
    def train(self, model_kwargs, train_kwargs):
        self.model = lightfm.LightFM(**model_kwargs)
        ones_interactions_matrix = self.interactions_matrix.copy()
        ones_interactions_matrix.data[:] = 1
        print('Training...')
        for i in tqdm(range(train_kwargs['num_epochs'])):
            self.model.fit_partial(interactions=ones_interactions_matrix, 
                                   user_features=train_kwargs['user_features'], 
                                   item_features=train_kwargs['item_features'], 
                                   sample_weight=self.interactions_matrix, 
                                   epochs=1, num_threads=train_kwargs['num_threads'], 
                                   verbose=False)
    
    def run(self, flat_interactions_df, lightfm_model, train_kwargs):
        self.preprocess(flat_interactions_df)
        self.train(lightfm_model, train_kwargs)