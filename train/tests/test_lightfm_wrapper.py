from pyrecs.train import lightfm_wrapper
import pytest
import pandas as pd
import numpy as np


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
def test_preprocess(df, interactions_type):
    lfm = lightfm_wrapper.LightFM(users_col='users', items_col='items', interactions_type=interactions_type,
                                  model_kwargs={}, train_kwargs={}, 
                                  n_recs=10, tfrs_prediction_batch_size=32)
    lfm.preprocess(train_df=df.copy(), 
                   test_df=pd.DataFrame({'users':[1],'items':[1]}),)
    interactions_matrix = lfm.interactions_matrix.todense()
    if interactions_type == 'ones':
        df.drop_duplicates(inplace=True)
        
    # Check that users have the right number of interactions
    user_counts = df.groupby('users').size().reset_index().rename(columns={0:'count'})
    user_counts['user_inds'] = user_counts['users'].map(lfm.user2ind)
    userind2count = dict(zip(user_counts['user_inds'], user_counts['count']))
    for user, count in userind2count.items():
        assert interactions_matrix[user].sum() == count
        
    # Check that items have the right number of interactions
    item_counts = df.groupby('items').size().reset_index().rename(columns={0:'count'})
    item_counts['item_inds'] = item_counts['items'].map(lfm.item2ind)
    itemind2count = dict(zip(item_counts['item_inds'], item_counts['count']))
    for item, count in itemind2count.items():
        assert interactions_matrix[:,item].sum() == count
        
    # Check that there are the right number of dimensions
    assert interactions_matrix.shape == (len(userind2count), len(itemind2count))