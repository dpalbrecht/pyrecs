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
def test_make_interactions(df, interactions_type):
    lfm = lightfm_wrapper.LightFM(rows_col='users', columns_col='items', interactions_type=interactions_type)
    lfm._make_interactions_matrix(df.copy())
    interactions_matrix = lfm.interactions_matrix.todense()
    if interactions_type == 'ones':
        df.drop_duplicates(inplace=True)
        
    # Check that rows have the right number of interactions
    row_counts = df.groupby('users').size().reset_index().rename(columns={0:'count'})
    row_counts['user_inds'] = row_counts['users'].map(lfm.rows2ind)
    rowind2count = dict(zip(row_counts['user_inds'], row_counts['count']))
    for row, count in rowind2count.items():
        assert interactions_matrix[row].sum() == count
        
    # Check that columns have the right number of interactions
    column_counts = df.groupby('items').size().reset_index().rename(columns={0:'count'})
    column_counts['item_inds'] = column_counts['items'].map(lfm.columns2ind)
    columnind2count = dict(zip(column_counts['item_inds'], column_counts['count']))
    for column, count in columnind2count.items():
        assert interactions_matrix[:,column].sum() == count
        
    # Check that there are the right number of dimensions
    assert interactions_matrix.shape == (len(rowind2count), len(columnind2count))