from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np
import itertools
from functools import partial


def _determine_feature_types(features_dict):
    feature_types = {'str':[],'list':[],'numeric':[],'array':[]}
    first_feature = list(features_dict.values())[0]
    for k, v in first_feature.items():
        if (isinstance(v, list)) and isinstance(v[0], str):
            feature_types['list'].append(k)
        elif isinstance(v, str):
            feature_types['str'].append(k)
        elif type(v) in [int, float]:
            feature_types['numeric'].append(k)
        elif isinstance(v, np.ndarray) and (v.dtype in [int, float]):
            feature_types['array'].append(k)
        else:
            raise ValueError(f"feature ('{k}') is not a supported type ('{v}')")
    return feature_types

def _format_features(features_dict, feature_types):
    feature_cols = feature_types['str']+feature_types['list']
    features_df = pd.DataFrame.from_dict(features_dict, orient='index').reset_index()
    if len(feature_cols) == 0:
        return features_df
    for col in feature_types['str']:
        features_df[col] = features_df[col].apply(lambda x: [f"{col}_{x}"])
    for col in feature_types['list']:
        features_df[col] = features_df[col].apply(lambda x: [f"{col}_{y}" for y in x])
    features_df['formatted_str_features'] = features_df[feature_cols].apply(lambda x: list(itertools.chain(*x.values)), axis=1)
    return features_df

def _create_feature2ind(features_df):
    if 'formatted_str_features' not in features_df.columns:
        return {}
    unique_str_features = set(itertools.chain(*features_df['formatted_str_features'].values))
    str_feature2ind = dict(zip(unique_str_features, range(len(unique_str_features))))
    return str_feature2ind

def _ohe_features(features, str_feature2ind):
    vector = np.zeros(len(str_feature2ind))
    for feature in features:
        feature_ind = str_feature2ind.get(feature)
        if feature_ind is not None:
            vector[feature_ind] = 1
    return vector.astype(int).tolist()

def _encode_features(features_df, str_feature2ind, feature_types):
    if len(str_feature2ind) > 0:
        feature_vectors = features_df['formatted_str_features'].apply(partial(_ohe_features, str_feature2ind=str_feature2ind))
    
    str_features = np.array([])
    if len(str_feature2ind) > 0:
        str_features = np.vstack(feature_vectors.values)
        
    numerical_features = np.array([])
    numerical_cols = feature_types['numeric']+feature_types['array']
    if len(numerical_cols) > 0:
        numerical_features = np.array([np.hstack(f) for f in features_df[numerical_cols].values])
        
    if (len(str_features) > 0) & (len(numerical_features) > 0):
        all_feature_vectors = np.hstack([str_features, numerical_features])
    elif len(str_features) > 0:
        all_feature_vectors = str_features
    else:
        all_feature_vectors = numerical_features
        
    id2featurevector = dict(zip(features_df['index'], all_feature_vectors))
    return id2featurevector

def _create_ordered_csr_matrix(id2featurevector, id2ind, normalize):
    features_matrix = []
    for data in sorted(list(id2ind.items()), key=lambda x: x[1]):
        id_, ind_ = data
        features_matrix.append(id2featurevector[id_])
    if normalize:
        features_matrix = np.divide(features_matrix, np.sum(features_matrix,axis=1).reshape(-1,1))
    features_matrix = csr_matrix(features_matrix)
    return features_matrix


def train_features_encoding(features_dict, id2ind, feature_type, normalize):
    # Determine feature types
    feature_types = _determine_feature_types(features_dict)
          
    # Format features
    features_df = _format_features(features_dict, feature_types)
    
    # Create feature 2 ind lookup
    str_feature2ind = _create_feature2ind(features_df)
    
    # Create id 2 feature lookup
    id2featurevector = _encode_features(features_df, str_feature2ind, feature_types)
    
    # Create ordered csr_matrix of features
    train_features_matrix = _create_ordered_csr_matrix(id2featurevector, id2ind, normalize)
    
    result = {
        f'train_{feature_type}_str_feature2ind':str_feature2ind, 
        f'train_{feature_type}_feature_types':feature_types, 
        f'train_{feature_type}_features_matrix':train_features_matrix
    }
    
    return result

def encode_new_features(features_dict, id2ind, feature_types, str_feature2ind, normalize):
     # Format features
    features_df = _format_features(features_dict, feature_types)
    
    # Create id 2 feature lookup
    id2featurevector = _encode_features(features_df, str_feature2ind, feature_types)
    
    # Create ordered csr_matrix of features
    features_matrix = _create_ordered_csr_matrix(id2featurevector, id2ind, normalize)
    
    return features_matrix