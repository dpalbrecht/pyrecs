import pytest
import numpy as np
from pyrecs.predict import tfrs_streaming
    

@pytest.mark.parametrize("inputs_dict,expected_predictions",
                         [
                             ({'user_identifiers':np.array(['user1', 'user2']), 
                               'user_embeddings':np.array([[1,2],[1,2]]), 
                               'item_identifiers':np.array(['a','b','c']), 
                               'item_embeddings':np.array([[1,2],[1,2],[1,2]]),
                               'train_users':['user1'],
                               'test_users':['user2'],
                               'n_recs':10},
                              {'train':{'user1': ['a', 'b', 'c']}, 
                               'test':{'user2': ['a', 'b', 'c']}}),
                             ({'user_identifiers':np.array(['user1', 'user2']), 
                               'user_embeddings':np.array([[1,0],[0,1]]), 
                               'item_identifiers':np.array(['a','b','c']), 
                               'item_embeddings':np.array([[1,0],[0,0],[0,1]]),
                               'train_users':['user1'],
                               'test_users':['user1','user2'],
                               'n_recs':1},
                              {'train':{'user1': ['a']}, 
                               'test':{'user1': ['a'], 'user2':['c']}})
                         ]
                        )
def test_predictions(inputs_dict, expected_predictions):
    predictions = tfrs_streaming.predict(**inputs_dict)
    assert predictions == expected_predictions
    

@pytest.mark.parametrize("inputs_dict",
                         [
                             ({'user_identifiers':np.array(['user1', 'user2']), 
                               'user_embeddings':np.array([[1,2],[1,2]]), 
                               'item_identifiers':np.array(['a','b','c']), 
                               'item_embeddings':np.array([[1,2],[1,2],[1,2]]),
                               'train_users':['user1'],
                               'test_users':[],
                               'n_recs':10})
                         ]
                        )
def test_unequal_user_ids(inputs_dict):
    with pytest.raises(Exception) as error:
        predictions = tfrs_streaming.predict(**inputs_dict)
    assert str(error.value) == 'All user identifiers must be in one or both of train/test users.'