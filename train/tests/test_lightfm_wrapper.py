from pyrecs.train import lightfm_wrapper
import pytest
import pandas as pd
import numpy as np
import itertools


def create_df_2users_3items_yes_duplicates(duplicates=True):
    df = pd.DataFrame({'users':['user1','user1','user1','user2'], 
                       'items':['item1','item1','item3','item2']})
    if not duplicates:
        return df.drop_duplicates()
    return df

def create_df_3users_2items_yes_duplicates(duplicates=True):
    df = pd.DataFrame({'users':['user1','user3','user1','user2'], 
                       'items':['item1','item1','item1','item2']})
    if not duplicates:
        return df.drop_duplicates()
    return df

def create_df_3users_3items_no_duplicates(duplicates=True):
    df = pd.DataFrame({'users':['user1','user2','user3','user3'], 
                       'items':['item1','item2','item3','item1']})
    if not duplicates:
        return df.drop_duplicates()
    return df

def create_df_3users_3items_yes_duplicates(duplicates=True):
    df = pd.DataFrame({'users':['user1','user2','user3','user3'], 
                       'items':['item1','item2','item3','item3']})
    if not duplicates:
        return df.drop_duplicates()
    return df

def format_test_parameters(funcs):
    parameters = []
    for func in funcs:
        parameters.extend([
            (func(), 'ones'),
            (func(), 'counts'),
            (func(duplicates=True), 'ones'),
            (func(duplicates=True), 'counts')
        ])
    return parameters

@pytest.mark.parametrize("df,interactions_type",
                         format_test_parameters([create_df_2users_3items_yes_duplicates, 
                                                 create_df_3users_2items_yes_duplicates, 
                                                 create_df_3users_3items_no_duplicates, 
                                                 create_df_3users_3items_yes_duplicates])
                        )
def test_create_train_interactions_matrix(df, interactions_type):
    lfm = lightfm_wrapper.LightFM(preprocess_kwargs={'interactions_type':interactions_type})
    lfm._create_train_interactions_matrix(train_df=df.copy(), 
                                          test_df=pd.DataFrame({'users':[1],'items':[1]}),)
    interactions_matrix = lfm.interactions_matrix.todense()
    if interactions_type == 'ones':
        df.drop_duplicates(inplace=True)
        
    # Check that users have the right number of interactions
    user_counts = df.groupby('users').size().reset_index().rename(columns={0:'count'})
    user_counts['user_inds'] = user_counts['users'].map(lfm.train_user2ind)
    userind2count = dict(zip(user_counts['user_inds'], user_counts['count']))
    for user, count in userind2count.items():
        assert interactions_matrix[user].sum() == count
        
    # Check that items have the right number of interactions
    item_counts = df.groupby('items').size().reset_index().rename(columns={0:'count'})
    item_counts['item_inds'] = item_counts['items'].map(lfm.train_item2ind)
    itemind2count = dict(zip(item_counts['item_inds'], item_counts['count']))
    for item, count in itemind2count.items():
        assert interactions_matrix[:,item].sum() == count
        
    # Check that there are the right number of dimensions
    assert interactions_matrix.shape == (len(userind2count), len(itemind2count))
    

@pytest.mark.parametrize("inputs,expected_error",
                         [
                             ({'train_df': pd.DataFrame({'users':[1,2,3],'items':[4,5,6]}),
                               'test_df': pd.DataFrame({'users':[1,2,4],'items':[4,5,7]}),
                               'train_user_features_dict': {},
                               'test_user_features_dict': {},
                               'train_item_features_dict': {},
                               'test_item_features_dict': {}},
                             'Train and Test, User and Item features are not string types.'),
                             ({'train_df': pd.DataFrame(),
                               'test_df': pd.DataFrame(),
                               'train_user_features_dict': {'user1':1},
                               'test_user_features_dict': {'user1':1},
                               'train_item_features_dict': {'item1':1},
                               'test_item_features_dict': {'item2':1}},
                             'Train and Test feature dictionaries are not mutually exclusive.'),
                             ({'train_df': pd.DataFrame({'users':['user1','user2'],'items':['item1','item1']}),
                               'test_df': pd.DataFrame(),
                               'train_user_features_dict': {'user1':1},
                               'test_user_features_dict': {'user2':1},
                               'train_item_features_dict': {'item1':1},
                               'test_item_features_dict': {'item2':1}},
                             'All Train Users do not have features.'),
                             ({'train_df': pd.DataFrame({'users':['user1','user2'],'items':['item1','item1']}),
                               'test_df': pd.DataFrame({'users':['user1'],'items':['item1']}),
                               'train_user_features_dict': {'user1':1,'user2':1},
                               'test_user_features_dict': {'user3':1},
                               'train_item_features_dict': {'item1':1},
                               'test_item_features_dict': {'item2':1}},
                             'All Users either do not have interactions or associated features.'),
                             ({'train_df': pd.DataFrame({'users':['user1','user2'],'items':['item1','item2']}),
                               'test_df': pd.DataFrame({'users':['user1'],'items':['item1']}),
                               'train_user_features_dict': {'user1':1,'user2':1},
                               'test_user_features_dict': {},
                               'train_item_features_dict': {'item1':1},
                               'test_item_features_dict': {'item2':1}},
                             'All Train Items do not have features.'),
                             ({'train_df': pd.DataFrame({'users':['user1','user2'],'items':['item1','item1']}),
                               'test_df': pd.DataFrame({'users':['user1'],'items':['item1']}),
                               'train_user_features_dict': {'user1':1,'user2':1},
                               'test_user_features_dict': {},
                               'train_item_features_dict': {'item1':1},
                               'test_item_features_dict': {'item2':1}},
                             'All Items either do not have interactions or associated features.')
                         ]) 
def test_quality_checks(inputs,expected_error):
    lfm = lightfm_wrapper.LightFM()
    with pytest.raises(Exception) as error:
        lfm._quality_checks(inputs['train_df'], inputs['test_df'],
                            inputs['train_user_features_dict'], inputs['train_item_features_dict'],
                            inputs['test_user_features_dict'], inputs['test_item_features_dict'])
    assert expected_error == str(error.value)


@pytest.mark.parametrize("inputs,expected_predictions",
                         [
                             ({
                               'postprocess_kwargs':{
                                   'remove_train_interactions_from_test':False,
                                   'fill_most_popular_items':False},
                               'train_df':pd.DataFrame({'users':['user1','user2'],'items':['item1','item2']}),
                               'test_df':pd.DataFrame({'users':['user1','user3'],'items':['item1','item3']})},
                              {'train':{'user1':['item1','item2'], 
                                        'user2':['item1','item2']},
                               'test':{'user1':['item1','item2'], 
                                       'user3':[]}}),
                             ({'postprocess_kwargs':{
                                   'remove_train_interactions_from_test':True,
                                   'fill_most_popular_items':False},
                               'train_df':pd.DataFrame({'users':['user1','user2'],'items':['item1','item2']}),
                               'test_df':pd.DataFrame({'users':['user1','user3'],'items':['item1','item3']})},
                              {'train':{'user1':['item1','item2'], 
                                        'user2':['item1','item2']},
                               'test':{'user1':['item2'], 
                                       'user3':[]}}),
                             ({'postprocess_kwargs':{
                                   'remove_train_interactions_from_test':True,
                                   'fill_most_popular_items':True},
                               'train_df':pd.DataFrame({'users':['user1','user2'],'items':['item1','item2']}),
                               'test_df':pd.DataFrame({'users':['user1','user3'],'items':['item1','item3']})},
                              {'train':{'user1':['item1','item2'], 
                                        'user2':['item1','item2']},
                               'test':{'user1':['item2'], 
                                       'user3':['item1','item2']}}),
                             ({'postprocess_kwargs':{
                                   'remove_train_interactions_from_test':False,
                                   'fill_most_popular_items':True},
                               'train_df':pd.DataFrame({'users':['user1','user2'],'items':['item1','item2']}),
                               'test_df':pd.DataFrame({'users':['user1','user3'],'items':['item1','item3']})},
                              {'train':{'user1':['item1','item2'], 
                                        'user2':['item1','item2']},
                               'test':{'user1':['item1','item2'], 
                                       'user3':['item1','item2']}})
                         ])
def test_no_side_features(inputs,expected_predictions):
    lfm = lightfm_wrapper.LightFM(model_kwargs={
                                  'no_components':10,
                                  'learning_rate':0.05,
                                  'loss':'warp',
                                  'random_state':42},
                                  train_kwargs={
                                   'num_epochs':2,
                                   'num_threads':1,
                                   'eval_epochs':'all',
                                   'plot':False},        
                                  postprocess_kwargs=inputs['postprocess_kwargs'])
    lfm.run(inputs['train_df'], inputs['test_df'], 
            train_user_features_dict={}, train_item_features_dict={},
            test_user_features_dict={}, test_item_features_dict={})
    lfm.predictions_dict = {k:{kk:sorted(vv) for kk,vv in v.items()} for k,v in lfm.predictions_dict.items()}
    assert lfm.predictions_dict == expected_predictions


def test_no_user_side_features():
    model_kwargs = {
        'no_components':10,
        'learning_rate':0.05,
        'loss':'warp',
        'random_state':42
    }
    train_kwargs = {
        'num_epochs':2,
        'num_threads':1,
        'eval_epochs':'all',
        'plot':False
    }
    lfm = lightfm_wrapper.LightFM(model_kwargs=model_kwargs, 
                                  train_kwargs=train_kwargs)
    train_df = pd.DataFrame({'users':['user1','user2'],'items':['item1','item2']})
    test_df = pd.DataFrame({'users':['user1','user3'],'items':['item1','item3']})
    train_user_features_dict = {}
    test_user_features_dict = {}
    train_item_features_dict = {'item1':{'type':'shirt'}, 'item2':{'type':'dress'}}
    test_item_features_dict = {'item3':{'type':'pants'}}
    lfm.run(train_df, test_df, 
            train_user_features_dict, train_item_features_dict,
            test_user_features_dict, test_item_features_dict)
    assert sorted(list(lfm.predictions_dict.keys())) == sorted(['user1','user2'])
    assert sorted(set(itertools.chain(*list(lfm.predictions_dict.values())))) == sorted(['item1','item2','item3'])
    

def test_no_item_side_features():
    model_kwargs = {
        'no_components':10,
        'learning_rate':0.05,
        'loss':'warp',
        'random_state':42
    }
    train_kwargs = {
        'num_epochs':2,
        'num_threads':1,
        'eval_epochs':'all',
        'plot':False
    }
    lfm = lightfm_wrapper.LightFM(model_kwargs=model_kwargs, 
                                  train_kwargs=train_kwargs)
    train_df = pd.DataFrame({'users':['user1','user2'],'items':['item1','item2']})
    test_df = pd.DataFrame({'users':['user1','user3'],'items':['item1','item3']})
    train_user_features_dict = {'user1':{'gender':'m'}, 'user2':{'gender':'f'}}
    test_user_features_dict = {'user3':{'gender':'m'}}
    train_item_features_dict = {}
    test_item_features_dict = {}
    lfm.run(train_df, test_df, 
            train_user_features_dict, train_item_features_dict,
            test_user_features_dict, test_item_features_dict)
    assert sorted(list(lfm.predictions_dict.keys())) == sorted(['user1','user2','user3'])
    assert sorted(set(itertools.chain(*list(lfm.predictions_dict.values())))) == sorted(['item1','item2'])
    
    
def test_full_side_features_small():
    model_kwargs = {
        'no_components':10,
        'learning_rate':0.05,
        'loss':'warp',
        'random_state':42
    }
    train_kwargs = {
        'num_epochs':2,
        'num_threads':1,
        'eval_epochs':'all',
        'plot':False
    }
    lfm = lightfm_wrapper.LightFM(model_kwargs=model_kwargs, 
                                  train_kwargs=train_kwargs)
    train_df = pd.DataFrame({'users':['user1','user2'],'items':['item1','item2']})
    test_df = pd.DataFrame({'users':['user1','user3'],'items':['item1','item3']})
    train_user_features_dict = {'user1':{'gender':'m'}, 'user2':{'gender':'f'}}
    test_user_features_dict = {'user3':{'gender':'m'}}
    train_item_features_dict = {'item1':{'type':'shirt'}, 'item2':{'type':'dress'}}
    test_item_features_dict = {'item3':{'type':'pants'}}
    lfm.run(train_df, test_df, 
            train_user_features_dict, train_item_features_dict,
            test_user_features_dict, test_item_features_dict)
    assert sorted(list(lfm.predictions_dict.keys())) == sorted(['user1','user2','user3'])
    assert sorted(set(itertools.chain(*list(lfm.predictions_dict.values())))) == sorted(['item1','item2','item3'])


def test_full_side_features_large():
    model_kwargs = {
        'no_components':10,
        'learning_rate':0.05,
        'loss':'warp',
        'random_state':42
    }
    train_kwargs = {
        'num_epochs':2,
        'num_threads':1,
        'eval_epochs':'all',
        'plot':False
    }
    lfm = lightfm_wrapper.LightFM(model_kwargs=model_kwargs, 
                                  train_kwargs=train_kwargs)
    train_df = pd.DataFrame({'users':['user1','user2','user1'],'items':['item1','item2','item2']})
    test_df = pd.DataFrame({'users':['user1','user3','user2'],'items':['item1','item3','item1']})
    train_user_features_dict = {'user1':{'gender':'m','favorite_products':['shirt','pants'], 'embedding':np.array([1,1,1])}, 
                                'user2':{'gender':'f', 'favorite_products':['shirt','pants', 'sweater'], 'embedding':np.array([2,2,2])}}
    test_user_features_dict = {'user3':{'gender':'m', 'favorite_products':[], 'embedding':np.array([3,3,3])}}
    train_item_features_dict = {'item1':{'type':'shirt', 'size':10}, 'item2':{'type':'dress', 'size':5}}
    test_item_features_dict = {'item3':{'type':'pants', 'size':32}}
    lfm.run(train_df, test_df, 
            train_user_features_dict, train_item_features_dict,
            test_user_features_dict, test_item_features_dict)
    assert sorted(list(lfm.predictions_dict.keys())) == sorted(['user1','user2','user3'])
    assert sorted(set(itertools.chain(*list(lfm.predictions_dict.values())))) == sorted(['item1','item2','item3'])