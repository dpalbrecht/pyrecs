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
    feature_types = mixed_features2vec._determine_feature_types(features_dict)
    assert expected_feature_types == feature_types

    
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

    
@pytest.mark.parametrize("features_df,expected_str_feature2ind",
                         [
                             (pd.DataFrame({'formatted_str_features':[['gender_m'],['gender_f'],
                                                                      ['gender_b'],['gender_g']]}),
                              {'gender_m':0,'gender_f':1,'gender_b':2,'gender_g':3}),
                             (pd.DataFrame({'gender':['m','f','b','g']}),
                              {})
                         ])
def test_create_feature2ind(features_df, expected_str_feature2ind):
    str_feature2ind = mixed_features2vec._create_feature2ind(features_df)
    assert sorted(list(expected_str_feature2ind.keys())) == sorted(list(str_feature2ind.keys()))
    assert sorted(list(expected_str_feature2ind.values())) == sorted(list(str_feature2ind.values()))


@pytest.mark.parametrize("features_dict,str_feature2ind,expected_id2featurevector",
                         [
                             (get_features_dict(),
                              {'gender_m':0,'gender_f':1,'favorite_products_shirt':2,'favorite_products_pants':3},
                              {'user1':np.array([1,0,1,1,21,1,2,3]),'user2':np.array([0,1,1,1,25,1,2,5])}),
                             (get_features_dict(exclude_cols=['gender','favorite_products']),
                              {},
                              {'user1':np.array([21,1,2,3]),'user2':np.array([25,1,2,5])}),
                             (get_features_dict(exclude_cols=['gender','favorite_products','age']),
                              {},
                              {'user1':np.array([1,2,3]),'user2':np.array([1,2,5])}),
                             (get_features_dict(exclude_cols=['gender','favorite_products','embedding']),
                              {},
                              {'user1':np.array([21]),'user2':np.array([25])}),
                             (get_features_dict(exclude_cols=['favorite_products','age','embedding']),
                              {'gender_m':0,'gender_f':1},
                              {'user1':np.array([1,0]),'user2':np.array([0,1])}),
                             (get_features_dict(exclude_cols=['gender','age','embedding']),
                              {'favorite_products_shirt':0,'favorite_products_pants':1},
                              {'user1':np.array([1,1]),'user2':np.array([1,1])})
                         ])
def test_encode_features(features_dict, str_feature2ind, expected_id2featurevector):
    feature_types = mixed_features2vec._determine_feature_types(features_dict)
    features_df = mixed_features2vec._format_features(features_dict, feature_types)
    id2featurevector = mixed_features2vec._encode_features(features_df, str_feature2ind, feature_types)
    np.testing.assert_equal(expected_id2featurevector, id2featurevector)


@pytest.mark.parametrize("id2featurevector,id2ind,expected_features_matrix",
                         [
                             ({'user1':np.array([1,2,3]),'user2':np.array([4,5,6])},
                              {'user1':0,'user2':1},
                              np.array([[1,2,3],[4,5,6]])),
                             ({'user1':np.array([1,2,3]),'user2':np.array([4,5,6])},
                              {'user1':1,'user2':0},
                              np.array([[4,5,6],[1,2,3]]))
                         ])
def test_create_ordered_csr_matrix(id2featurevector, id2ind, expected_features_matrix):
    features_matrix = mixed_features2vec._create_ordered_csr_matrix(id2featurevector, id2ind, normalize=False).todense()
    assert (expected_features_matrix == features_matrix).all()
    
def test_create_ordered_csr_matrix_normalize():
    id2featurevector = {'user1':np.array([1,2,3]),'user2':np.array([4,5,6])}
    id2ind = {'user1':0,'user2':1}
    expected_features_matrix = np.array([[1,2,3],[4,5,6]])
    features_matrix = mixed_features2vec._create_ordered_csr_matrix(id2featurevector, id2ind, normalize=True).todense()
    assert set(np.asarray(np.sum(features_matrix, axis=1)).flatten().tolist()) == set([1])