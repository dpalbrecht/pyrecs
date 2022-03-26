import pytest
import numpy as np
from pyrecs.predict import tfrs_streaming
    

@pytest.mark.parametrize("inputs_dict,expected_predictions",
                         [
                             ({'user_identifiers':np.array(['user1', 'user2']), 
                               'user_embeddings':np.array([[1,2],[1,5]]), 
                               'item_identifiers':np.array(['a','b','c']), 
                               'item_embeddings':np.array([[1,2],[1,2],[1,2]]),
                               'train_user_interactions':{},
                               'n_recs':10},
                              {'train':{}, 
                               'test':{'user1': ['a', 'b', 'c'], 
                                       'user2': ['a', 'b', 'c']}}),
                             ({'user_identifiers':np.array(['user1', 'user2']), 
                               'user_embeddings':np.array([[1,0],[0,1]]), 
                               'item_identifiers':np.array(['a','b','c']), 
                               'item_embeddings':np.array([[1,0],[0,0],[0,1]]),
                               'train_user_interactions':{},
                               'n_recs':1},
                              {'train':{}, 
                               'test':{'user1': ['a'], 
                                       'user2': ['c']}}),
                             ({'user_identifiers':np.array(['user1', 'user2']), 
                               'user_embeddings':np.array([[1,0],[0,1]]), 
                               'item_identifiers':np.array(['a','b','c']), 
                               'item_embeddings':np.array([[1,0],[0,0],[0,1]]),
                               'train_user_interactions':{'user1':['a']},
                               'n_recs':1},
                              {'train':{'user1': ['a']}, 
                               'test':{'user1': ['b'], 
                                       'user2': ['c']}})
                         ]
                        )
def test_predictions(inputs_dict, expected_predictions):
    predictions = tfrs_streaming.predict(user_identifiers=inputs_dict['user_identifiers'], 
                                         user_embeddings=inputs_dict['user_embeddings'],
                                         item_identifiers=inputs_dict['item_identifiers'], 
                                         item_embeddings=inputs_dict['item_embeddings'],
                                         n_recs=inputs_dict['n_recs'],
                                         train_user_interactions=inputs_dict['train_user_interactions'])
    assert predictions == expected_predictions