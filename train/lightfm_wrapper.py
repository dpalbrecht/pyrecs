import lightfm
from scipy.sparse import coo_matrix, csr_matrix, vstack
import numpy as np
import pandas as pd
import itertools
from functools import partial
from tqdm import tqdm
import matplotlib.pyplot as plt
from pyrecs.evaluate.mapk import mapk
from pyrecs.predict import tfrs_streaming


def ohe_features(features, lookup):
    vector = np.zeros(len(lookup))
    for feature in features:
        feature_ind = lookup.get(feature)
        if feature_ind is not None:
            vector[feature_ind] = 1
    return csr_matrix(vector.astype(int))

def format_features(features_dict, str_features, list_features):
    features_df = pd.DataFrame(features_dict).T.reset_index()
    for col in str_features:
        features_df[col] = features_df[col].apply(lambda x: [f"{col}_{x}"])
    for col in list_features:
        features_df[col] = features_df[col].apply(lambda x: [f"{col}_{y}" for y in x])
    features_df['formatted_features'] = features_df[str_features+list_features].apply(lambda x: list(itertools.chain(*x.values)), axis=1)
    return features_df

def encode_features(features_df, feature2ind):
    feature_vectors = features_df['formatted_features'].apply(partial(ohe_features, lookup=feature2ind))
    id2featurevector = dict(zip(features_df['index'], feature_vectors))
    return id2featurevector

def train_features_lookup(features_dict, id2ind, feature_type):
    # Determine feature types
    str_features, list_features = [], []
    first_feature = list(features_dict.values())[0]
    for k, v in first_feature.items():
        if isinstance(v, list):
            list_features.append(k)
        elif isinstance(v, str) | (type(v) in [int, float]):
            str_features.append(k)
        else:
            raise ValueError(f"feature ('{k}') is not a supported type ('{v}')")
          
    # Format features
    features_df = format_features(features_dict, str_features, list_features)
    
    # Create feature 2 ind lookup
    unique_features = set(itertools.chain(*features_df['formatted_features'].values))
    feature2ind = dict(zip(unique_features, range(len(unique_features))))
    
    # Create id 2 feature lookup
    id2featurevector = encode_features(features_df, feature2ind)
    
    # Create ordered csr_matrix of features
    train_features_matrix = []
    for data in sorted(list(id2ind.items()), key=lambda x: x[1], reverse=True):
        id_, ind_ = data
        train_features_matrix.append(id2featurevector[id_])
    train_features_matrix = vstack(train_features_matrix)
    
    result = {
        f'{feature_type}_train_feature2ind':feature2ind, 
        f'{feature_type}_train_id2featurevector':id2featurevector, 
        f'{feature_type}_train_str_features':str_features, 
        f'{feature_type}_train_list_features':list_features,
        f'{feature_type}_train_features_matrix':train_features_matrix
    }
    
    return result

def encode_new_features(features_dict, id2ind, str_features, list_features, feature2ind):
     # Format features
    features_df = format_features(features_dict, str_features, list_features)
    
    # Create id 2 feature lookup
    id2featurevector = encode_features(features_df, feature2ind)
    
    # Create ordered csr_matrix of features
    features_matrix = []
    for data in sorted(list(id2ind.items()), key=lambda x: x[1], reverse=True):
        id_, ind_ = data
        features_matrix.append(id2featurevector[id_])
    features_matrix = vstack(features_matrix)
    
    return features_matrix


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
        
    def preprocess(self, train_df, test_df, 
                   train_user_features_dict, train_item_features_dict,
                   test_user_features_dict, test_item_features_dict):
        
        # Quality checks
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
        
        # Format train and test dfs
        self.train_dict = dict(train_df.groupby(self.users_col)[self.items_col].apply(lambda x: list(x.unique())))
        self.test_dict = dict(test_df.groupby(self.users_col)[self.items_col].apply(lambda x: list(x.unique())))
        
        # Create training interactions matrix
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
        
        # Process train set user/item features
        self.user_train_features_matrix, self.item_train_features_matrix = None, None
        if len(train_user_features_dict) > 0:
            feature_lookups = train_features_lookup(train_user_features_dict, 
                                                    id2ind=self.train_user2ind, feature_type='user')
            self.__dict__.update(feature_lookups)
        if len(train_item_features_dict) > 0:
            feature_lookups = train_features_lookup(train_item_features_dict, 
                                                    id2ind=self.train_item2ind, feature_type='item')
            self.__dict__.update(feature_lookups)
        feature_lookups = None
                
        # Process test set user/item features
        self.user_test_id2featurevector, self.item_test_id2featurevector = None, None
        self.test_user2ind, self.test_item2ind = {}, {}
        if len(test_user_features_dict) > 0:
            self.test_user2ind = dict(zip(test_user_features_dict.keys(),
                                      range(len(self.train_user2ind), 
                                            len(test_user_features_dict.keys())+len(self.train_user2ind))))
            self.user_test_features_matrix = encode_new_features(test_user_features_dict, 
                                                                 self.test_user2ind,
                                                                 self.user_train_str_features, 
                                                                 self.user_train_list_features, 
                                                                 self.user_train_feature2ind)
        if len(test_item_features_dict) > 0:
            self.test_item2ind = dict(zip(test_item_features_dict.keys(),
                                      range(len(self.test_item2ind), 
                                            len(test_item_features_dict.keys())+len(self.train_item2ind))))
            self.item_test_features_matrix = encode_new_features(test_item_features_dict, 
                                                                 self.test_item2ind,
                                                                 self.item_train_str_features, 
                                                                 self.item_train_list_features, 
                                                                 self.item_train_feature2ind)
        
        # Make sure n_recs isn't > number of items available
        self.n_recs = min(self.n_recs, len(self.train_item2ind))

    def evaluate(self):       
        # Format user/item representations and identifiers in train
        train_user_biases, train_user_factors = self.model.get_user_representations(features=self.user_train_features_matrix)
        train_user_factors = np.concatenate((train_user_factors, np.ones((train_user_biases.shape[0], 1))), axis=1)
        train_user_identifiers = [r[0] for r in sorted(self.train_user2ind.items(), key=lambda x: x[1])]
        train_item_biases, train_item_factors = self.model.get_item_representations(features=self.item_train_features_matrix)
        train_item_factors = np.concatenate((train_item_factors, train_item_biases.reshape(-1, 1)), axis=1)
        train_item_identifiers = [c[0] for c in sorted(self.train_item2ind.items(), key=lambda x: x[1])]
        
        # Add user/item representations and identifiers from test
        test_user_biases, test_user_factors = self.model.get_user_representations(features=self.user_test_features_matrix)
        test_user_factors = np.concatenate((test_user_factors, np.ones((test_user_biases.shape[0], 1))), axis=1)
        test_user_identifiers = [r[0] for r in sorted(self.test_user2ind.items(), key=lambda x: x[1])]
        test_item_biases, test_item_factors = self.model.get_item_representations(features=self.item_test_features_matrix)
        test_item_factors = np.concatenate((test_item_factors, np.ones((test_item_biases.shape[0], 1))), axis=1)
        test_item_identifiers = [c[0] for c in sorted(self.test_item2ind.items(), key=lambda x: x[1])]
        
        # Combine train+test user/item representations
        user_factors = np.vstack([train_user_factors, test_user_factors])
        item_factors = np.vstack([train_item_factors, test_item_factors])
        user_identifiers = train_user_identifiers+test_user_identifiers
        item_identifiers = train_item_identifiers+test_item_identifiers
        
        # Predict
        # TODO: Add functionality to filter out items that users have already interacted with in train
        # TODO: Predict most popular items for new users, where no cold-start
        predictions = tfrs_streaming.predict(user_identifiers=user_identifiers, 
                                             user_embeddings=user_factors,
                                             item_identifiers=item_identifiers, 
                                             item_embeddings=item_factors,
                                             embedding_dtype='float32', 
                                             n_recs=self.n_recs,
                                             prediction_batch_size=self.tfrs_prediction_batch_size)
        
        # Format train set predictions
        train_set_truth, train_set_predictions = [], []
        for user, truth in self.train_dict.items():
            train_set_predictions.append(predictions[user])
            train_set_truth.append(truth)
        
        # Format test set predictions
        test_set_truth, test_set_predictions = [], []
        for user, truth in self.test_dict.items():
            test_set_predictions.append(predictions[user])
            test_set_truth.append(truth)
            
        # Calculate MAP@K
        train_mapk = mapk(train_set_truth, train_set_predictions, k=self.n_recs)
        test_mapk = mapk(test_set_truth, test_set_predictions, k=self.n_recs)

        return train_mapk, test_mapk

    def train(self):
        self.model = lightfm.LightFM(**self.model_kwargs)
        ones_interactions_matrix = self.interactions_matrix.copy()
        ones_interactions_matrix.data[:] = 1
        self.train_evaluations, self.test_evaluations, self.eval_epochs = [], [], []
        for epoch in tqdm(range(1, self.train_kwargs['num_epochs']+1), desc='Training', leave=True, position=0):
            self.model.fit_partial(interactions=ones_interactions_matrix, 
                                   user_features=self.user_train_features_matrix, 
                                   item_features=self.item_train_features_matrix, 
                                   sample_weight=self.interactions_matrix, 
                                   epochs=1, num_threads=self.train_kwargs['num_threads'], 
                                   verbose=False)
            if (self.train_kwargs['eval_epochs'] == 'all') or (epoch in self.train_kwargs['eval_epochs']):
                train_mapk, test_mapk = self.evaluate()
                self.eval_epochs.append(epoch)
                self.train_evaluations.append(train_mapk)
                self.test_evaluations.append(test_mapk)
        plt.plot(self.eval_epochs, self.train_evaluations, linestyle='dotted')
        plt.scatter(self.eval_epochs, self.train_evaluations, label='Train')
        plt.plot(self.eval_epochs, self.test_evaluations, linestyle='dotted')
        plt.scatter(self.eval_epochs, self.test_evaluations, label='Test')
        plt.xticks(self.eval_epochs)
        plt.yticks(np.arange(0,1.1,.1))
        plt.ylim(-0.05,1.05)
        plt.legend(loc=(1.01,0))
        plt.title('Evaluation')
        plt.xlabel('Epoch')
        plt.ylabel(f'MAP@{self.n_recs}')
    
    def run(self, train_df, test_df, 
            train_user_features_dict, train_item_features_dict,
            test_user_features_dict, test_item_features_dict):
        self.preprocess(train_df, test_df, train_user_features_dict, train_item_features_dict, 
                        test_user_features_dict, test_item_features_dict)
        self.train()