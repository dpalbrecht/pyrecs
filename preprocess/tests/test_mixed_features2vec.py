from pyrecs.preprocess import mixed_features2vec
import pytest
import pandas as pd
import numpy as np


def get_features_dict(exclude_cols=[]):
    features_dict = {
    'user1':{
        'age':21,
        'gender':'m',
        'favorite_products':['shirt','pants'],
        'embedding':np.array([1,2,3])
    },
    'user2':{
        'age':25,
        'gender':'f',
        'favorite_products':['shirt','pants'],
        'embedding':np.array([1,2,5])
        }
    }
    for k, v in features_dict.items():
        for col in exclude_cols:
            del v[col]
    return features_dict


@pytest.mark.parametrize("features_dict,expected_feature_types",
                         [
                             (get_features_dict(),
                              {'str':['gender'],'list':['favorite_products'],
                               'numeric':['age'],'array':['embedding']}),
                             (get_features_dict(exclude_cols=['age', 'favorite_products']),
                              {'str':['gender'],'list':[],
                               'numeric':[],'array':['embedding']})
                         ])
def test_determine_feature_types(features_dict, expected_feature_types):
    assert expected_feature_types == mixed_features2vec._determine_feature_types(features_dict)

    
@pytest.mark.parametrize("features_dict,expected_features_df",
                         [
                             (get_features_dict(exclude_cols=['age','favorite_products','embedding']),
                              pd.DataFrame(
                                {'index':['user1','user2'],
                                 'gender':[['gender_m'],['gender_f']],
                                 'formatted_str_features':[['gender_m'],['gender_f']]}
                             )),
                             (get_features_dict(exclude_cols=['favorite_products']),
                              pd.DataFrame(
                                {'index':['user1','user2'],
                                 'gender':[['gender_m'],['gender_f']],
                                 'age':[21,25],
                                 'embedding':[np.array([1,2,3]),np.array([1,2,5])],
                                 'formatted_str_features':[['gender_m'],['gender_f']]}
                             ))
                         ])
def test_format_features(features_dict, expected_features_df):
    feature_types = mixed_features2vec._determine_feature_types(features_dict)
    features_df = mixed_features2vec._format_features(features_dict, feature_types)
    assert sorted(features_df.columns) == sorted(expected_features_df.columns)
    assert expected_features_df[features_df.columns].sort_values(by='index').equals(features_df.sort_values(by='index'))

    
def test_create_feature2ind():
    pass

def test_encode_features():
    pass

def test_create_ordered_csr_matrix():
    pass

def test_ohe_features():
    pass



