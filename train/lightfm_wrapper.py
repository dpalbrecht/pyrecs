import lightfm
from scipy.sparse import coo_matrix, csr_matrix, vstack
import numpy as np
import pandas as pd
import itertools
from functools import partial
from tqdm import tqdm
import matplotlib.pyplot as plt
from pyrecs.evaluate.mapk import mapk
from IPython.display import clear_output
from pyrecs.predict import tfrs_streaming
from pyrecs.preprocess import mixed_features2vec


# TODO: Clean up model inputs here and in tests
class LightFM:
    def __init__(self, 
                 users_col='users', items_col='items', interactions_type='ones', 
                 model_kwargs={}, train_kwargs={},
                 fill_most_popular=False, normalize_features=False,
                 remove_factor_biases=False,
                 n_recs=10, tfrs_prediction_batch_size=32):
        self.users_col = users_col
        self.items_col = items_col
        self.interactions_type = interactions_type
        self.model_kwargs = model_kwargs
        self.train_kwargs = train_kwargs
        self.n_recs = n_recs
        self.tfrs_prediction_batch_size = tfrs_prediction_batch_size
        self.fill_most_popular = fill_most_popular
        self.normalize_features = normalize_features
        self.remove_factor_biases = remove_factor_biases
        self.train_user_features_matrix, self.train_item_features_matrix = None, None
        self.test_user_features_matrix, self.test_item_features_matrix = None, None
        self.user_test_id2featurevector, self.item_test_id2featurevector = None, None
        self.test_user2ind, self.test_item2ind = {}, {}
        
    def _quality_checks(self, train_df, test_df,
                        train_user_features_dict, train_item_features_dict,
                        test_user_features_dict, test_item_features_dict):
        try:
            if len(train_df) > 0:
                assert isinstance(train_df[self.users_col].iloc[0], str)
                assert isinstance(train_df[self.items_col].iloc[0], str)
            if len(test_df) > 0:
                assert isinstance(test_df[self.users_col].iloc[0], str)
                assert isinstance(test_df[self.items_col].iloc[0], str)
        except:
            raise AssertionError('Train and Test, User and Item features are not string types.')
        try:
            assert len(set(train_user_features_dict.keys()) & set(test_user_features_dict.keys())) == 0
            assert len(set(train_item_features_dict.keys()) & set(test_item_features_dict.keys())) == 0
        except:
            raise AssertionError('Train and Test feature dictionaries are not mutually exclusive.')
        if len(train_user_features_dict) > 0:
            try:
                assert set(train_df[self.users_col]) == set(train_user_features_dict.keys())
            except:
                raise AssertionError('All Train Users do not have features.')
            try:
                all_users_interactions = train_df[self.users_col].unique().tolist()+test_df[self.users_col].unique().tolist()
                all_users_features = list(train_user_features_dict.keys())+list(test_user_features_dict.keys())
                assert set(all_users_interactions) == set(all_users_features)
            except:
                raise AssertionError('All Users either do not have interactions or associated features.')
        if len(train_item_features_dict) > 0:
            try:
                assert set(train_df[self.items_col]) == set(train_item_features_dict.keys())
            except:
                raise AssertionError('All Train Items do not have features.')
            try:
                all_items_interactions = train_df[self.items_col].unique().tolist()+test_df[self.items_col].unique().tolist()
                all_items_features = list(train_item_features_dict.keys())+list(test_item_features_dict.keys())
                assert set(all_items_interactions) == set(all_items_features)
            except:
                raise AssertionError('All Items either do not have interactions or associated features.')
                
    def _format_dfs(self, df):
        return dict(df.groupby(self.users_col)[self.items_col].apply(lambda x: list(x.unique())))
    
    # TODO: Be able to use interaction ratings as weights (new interaction_type)
    def _create_train_interactions_matrix(self, train_df, test_df):
        self.train_user2ind = dict(zip(train_df[self.users_col].unique(),
                                       range(train_df[self.users_col].nunique())))
        self.train_item2ind = dict(zip(train_df[self.items_col].unique(),
                                       range(train_df[self.items_col].nunique())))
        if self.interactions_type == 'ones':
            train_df.drop_duplicates(subset=[self.users_col, self.items_col], inplace=True)
            users = train_df[self.users_col].map(self.train_user2ind).values
            items = train_df[self.items_col].map(self.train_item2ind).values
            interactions = np.ones(len(train_df))
        elif self.interactions_type == 'counts':
            counts_dict = dict(train_df.groupby([self.users_col, self.items_col]).size())
            unique_user_item_pairs = train_df.groupby(self.users_col)[self.items_col].unique()\
                                            .reset_index().explode(self.items_col).values
            users, items, interactions = [], [], []
            for unique_pair in unique_user_item_pairs:
                user, item = unique_pair
                users.append(self.train_user2ind[user])
                items.append(self.train_item2ind[item])
                interactions.append(counts_dict[(user, item)])
        else:
            raise ValueError(f"interactions_type ('{self.interactions_type}') not implemented.")
        self.interactions_matrix = coo_matrix((interactions, (users, items)), 
                                              shape=(len(self.train_user2ind), len(self.train_item2ind)))
        
    def _process_train_features(self, train_features_dict, feature_type):
        if len(train_features_dict) > 0:
            feature_encodings = mixed_features2vec.train_features_encoding(train_features_dict, 
                                                                           id2ind=self.__dict__[f'train_{feature_type}2ind'], 
                                                                           feature_type=feature_type,
                                                                           normalize=self.normalize_features)
            self.__dict__.update(feature_encodings)
            
    def _process_test_features(self, test_features_dict, feature_type):
        if len(test_features_dict) > 0:
            num_train_keys = len(self.__dict__[f'train_{feature_type}2ind'])
            self.__dict__[f'test_{feature_type}2ind'] = dict(zip(test_features_dict.keys(),
                                                                 range(num_train_keys, len(test_features_dict.keys())+num_train_keys)))
            self.__dict__[f'test_{feature_type}_features_matrix'] = mixed_features2vec.encode_new_features(test_features_dict,
                                                                                            self.__dict__[f'test_{feature_type}2ind'], 
                                                                                            self.__dict__[f'train_{feature_type}_feature_types'],
                                                                                            self.__dict__[f'train_{feature_type}_str_feature2ind'],
                                                                                            normalize=self.normalize_features)
        
    def preprocess(self, train_df, test_df, 
                   train_user_features_dict, train_item_features_dict,
                   test_user_features_dict, test_item_features_dict):
        
        # Input quality checks
        self._quality_checks(train_df, test_df, 
                             train_user_features_dict, train_item_features_dict,
                             test_user_features_dict, test_item_features_dict)
        
        # Format train and test dfs
        self.train_dict = self._format_dfs(train_df)
        self.test_dict = self._format_dfs(test_df)
        
        # Create train interactions matrix
        self._create_train_interactions_matrix(train_df, test_df)
        
        # Process train set user/item features
        self._process_train_features(train_user_features_dict, feature_type='user')
        self._process_train_features(train_item_features_dict, feature_type='item')
                
        # Process test set user/item features
        self._process_test_features(test_user_features_dict, feature_type='user')
        self._process_test_features(test_item_features_dict, feature_type='item')
        
        # Make sure n_recs isn't > number of items available
        unique_item_ids = set(train_df[self.items_col].unique().tolist() + test_df[self.items_col].unique().tolist())
        self.n_recs = min(self.n_recs, len(unique_item_ids))
        
        # Get most popular items
        if self.fill_most_popular:
            self.most_popular_items = train_df[self.items_col].value_counts().head(self.n_recs).index.tolist()
    
    def _get_feature_representations(self, dataset, feature_type):
        if feature_type == 'user':
            biases, factors = self.model.get_user_representations(features=self.__dict__[f'{dataset}_user_features_matrix'])
            if not self.remove_factor_biases:
                factors = np.concatenate((factors, np.ones((biases.shape[0], 1))), axis=1)
        else:
            biases, factors = self.model.get_item_representations(features=self.__dict__[f'{dataset}_item_features_matrix'])
            if not self.remove_factor_biases:
                factors = np.concatenate((factors, biases.reshape(-1, 1)), axis=1)
        identifiers = [u[0] for u in sorted(self.__dict__[f'{dataset}_{feature_type}2ind'].items(), key=lambda x: x[1])]
        return factors, identifiers
    
    def _format_predictions(self, predictions_dict, dataset):
        formatted_truths, formatted_predictions = [], []
        for user, truth in self.__dict__[f'{dataset}_dict'].items():
            formatted_truths.append(truth)
            predictions = predictions_dict.get(user, [])
            if self.fill_most_popular:
                while len(predictions) < self.n_recs:
                    for i in self.most_popular_items:
                        if i not in predictions:
                            predictions.append(i)
            formatted_predictions.append(predictions)
        return formatted_truths, formatted_predictions
        
    # TODO: Add analysis for novelty and diversity of recommendations
    # TODO: Add functionality to filter out items that users have already interacted with in train
    def evaluate(self, save_predictions):       
        # Format train/test user/item representations and identifiers
        train_user_factors, train_user_identifiers = self._get_feature_representations(dataset='train', feature_type='user')
        train_item_factors, train_item_identifiers = self._get_feature_representations(dataset='train', feature_type='item')
        test_item_identifiers, test_user_identifiers = [], []
        if self.test_user_features_matrix is not None:
            test_user_factors, test_user_identifiers = self._get_feature_representations(dataset='test', feature_type='user')
            user_embeddings = np.vstack([train_user_factors, test_user_factors])
        else:
            user_embeddings = train_user_factors
        if self.test_item_features_matrix is not None:
            test_item_factors, test_item_identifiers = self._get_feature_representations(dataset='test', feature_type='item')
            item_embeddings = np.vstack([train_item_factors, test_item_factors])
        else:
            item_embeddings = train_item_factors
        user_identifiers = train_user_identifiers+test_user_identifiers
        item_identifiers = train_item_identifiers+test_item_identifiers
        
        # Predict
        predictions_dict = tfrs_streaming.predict(user_identifiers=user_identifiers, 
                                                  user_embeddings=user_embeddings,
                                                  item_identifiers=item_identifiers, 
                                                  item_embeddings=item_embeddings,
                                                  embedding_dtype='float32', 
                                                  n_recs=self.n_recs,
                                                  prediction_batch_size=self.tfrs_prediction_batch_size)
        if save_predictions:
            self.predictions_dict = predictions_dict
        
        # Format predictions
        train_truth, train_predictions = self._format_predictions(predictions_dict, dataset='train')
        test_truth, test_predictions = self._format_predictions(predictions_dict, dataset='test')
            
        # Calculate MAP@K
        train_mapk = mapk(train_truth, train_predictions, k=self.n_recs)
        test_mapk = mapk(test_truth, test_predictions, k=self.n_recs)

        return train_mapk, test_mapk

    # TODO: Plot cold-start test MAP@K separately than warm-start, and then plot them together
    def train(self):
        self.model = lightfm.LightFM(**self.model_kwargs)
        ones_interactions_matrix = self.interactions_matrix.copy()
        ones_interactions_matrix.data[:] = 1
        self.train_evaluations, self.test_evaluations, self.eval_epochs = [], [], []
        for epoch in tqdm(range(1, self.train_kwargs['num_epochs']+1), desc='Training', position=0):
            self.model.fit_partial(interactions=ones_interactions_matrix, 
                                   user_features=self.train_user_features_matrix, 
                                   item_features=self.train_item_features_matrix, 
                                   sample_weight=self.interactions_matrix, 
                                   epochs=1, num_threads=self.train_kwargs['num_threads'], 
                                   verbose=False)
            if (self.train_kwargs['eval_epochs'] == 'all') or (epoch in self.train_kwargs['eval_epochs']):
                # Evaluate results
                train_mapk, test_mapk = self.evaluate(save_predictions = (epoch==self.train_kwargs['num_epochs']))
                self.eval_epochs.append(epoch)
                self.train_evaluations.append(train_mapk)
                self.test_evaluations.append(test_mapk)
                
                clear_output()
                if self.train_kwargs['plot']:
                    # Plot results
                    plt.plot(self.eval_epochs, self.train_evaluations, linestyle='dotted')
                    plt.scatter(self.eval_epochs, self.train_evaluations, label='Train')
                    plt.plot(self.eval_epochs, self.test_evaluations, linestyle='dotted')
                    plt.scatter(self.eval_epochs, self.test_evaluations, label='Test')
                    for eval_list in [self.train_evaluations, self.test_evaluations]:
                        for n, e in enumerate(eval_list):
                            plt.text(n+1, e, f"{e:.3f}", ha='center', va='bottom')
                    plt.xticks(self.eval_epochs)
                    plt.legend(loc=(1.01,0))
                    plt.title('Evaluation')
                    plt.xlabel('Epoch')
                    plt.ylabel(f'MAP@{self.n_recs}')
                    plt.show();
    
    def run(self, 
            train_df, test_df, 
            train_user_features_dict={}, train_item_features_dict={},
            test_user_features_dict={}, test_item_features_dict={}):
        self.preprocess(train_df, test_df, train_user_features_dict, train_item_features_dict, 
                        test_user_features_dict, test_item_features_dict)
        self.train()